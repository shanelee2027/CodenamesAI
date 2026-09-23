"""The distilled listener's training machinery: loading teacher rankings into
positions, turning positions into Plackett-Luce choice groups, the
group-softmax objective, and the fit metrics. Used by
scripts/pipeline/train_listener.py and by every tool that evaluates or tunes
the listener, which is why it lives in the package rather than in a script.

The formulation is in scripts/pipeline/train_listener.py's docstring and
codenames/listener_features.py.
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from codenames.board import Board
from codenames.clue_stats import ClueStats
from codenames.listener_features import (
    FEATURE_NAMES, EntitySims, ExtraSims, SwowTables, WordNorms, WordStats,
    extract,
)
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

CACHE = PROJECT_ROOT / "cache"
DB = CACHE / "llm_store.db"
WORD_STATS = CACHE / "word_stats.npz"
SWOW_TABLES = CACHE / "swow.npz"
ENTITY_SIMS = CACHE / "entity_sims.npz"
LM_PMI = CACHE / "lm_pmi.npz"
EXTRA_SIMS = CACHE / "extra_sims.npz"
WORD_NORMS = CACHE / "word_norms.npz"
WORDNET_SIMS = CACHE / "wordnet_sims.npz"
LEXICAL_SIMS = CACHE / "lexical_sims.npz"
DEFAULT_MODEL = "claude-sonnet-5+effort=medium"

# Named feature blocks, so an ablation is a flag rather than an edit. The point
# of `nb` is that it is EXACTLY what a numberbatch-only model can see: the three
# numberbatch views, plus every derived feature that happens to be computed from
# the numberbatch column alone (peak_z, lead_margin, p_max_*, word_*, cohesion_*
# all index space 1). Anything reading glove, wiki2vec, SWOW or entity is out.
FEATURE_BLOCKS: dict[str, list[str]] = {
    "nb": ["z_numberbatch", "gaptop_numberbatch",
           "k", "n_candidates", "peak_z", "lead_margin", "p_max_sigma2",
           "word_mean_sim", "word_sd_sim", "cohesion", "cohesion_minus_own"],
    "spaces": ["z_glove", "gaptop_glove", "gaptop_wiki2vec", "own_min_space"],
    "swow": ["swow1", "swow2", "swow_has"],
    "swowrev": ["swow_rev1", "swow_rev2", "swow_rev2_share", "swow_asym"],
    "entity": ["ent_rank", "ent_has"],
    "pmi": ["pmi_gaptop"],
    "extraspaces": ["g840_z", "g840_rank", "g840_gaptop", "ft_z", "ft_gaptop"],
    "norms": ["conc", "conc_sd", "pct_known", "log_freq", "n_senses"],
    "wordnet": ["wn_wup", "wn_wup_rank"],
    "lexical": ["orth_contains", "orth_prefix", "orth_suffix", "orth_trigram",
                "gloss_c_in_w", "gloss_w_in_c", "gloss_jaccard"],
}
assert sorted(sum(FEATURE_BLOCKS.values(), [])) == sorted(FEATURE_NAMES), "blocks must partition FEATURE_NAMES"


def board_lookup(max_seed: int, collected: int = 0) -> list[tuple[int, frozenset[str]]]:
    """(seed, word set) for every board a cached response could have come from.

    Two sources, built with different vocabularies and disjoint seed ranges:
    arena games use `Board.generate(seed)` over all 400 board words, while
    scripts/data/collect_listener_data.py generates from the 250-word TRAINING
    list at seeds >= 1e6 (the holdout guard -- see that script). Both must be
    searched or one source silently resolves to nothing and is dropped whole.
    """
    from codenames.board import load_training_wordlist

    out = [(s, frozenset(w.lower() for w in Board.generate(seed=s).words)) for s in range(max_seed)]
    if collected:
        vocab = load_training_wordlist()
        base = 1_000_000
        out += [(base + i, frozenset(w.lower() for w in Board.generate(seed=base + i, vocabulary=vocab).words))
                for i in range(collected)]
    return out


def resolve_seed(candidates: list[str], boards: list[tuple[int, frozenset[str]]]) -> int | None:
    """Which board this position came from, or None if it is ambiguous.

    A position is a subset of its board, so the board is the one whose word set
    contains every candidate. Two boards sharing a 16-word subset is vanishingly
    unlikely (they are 25-word samples from 400), but ambiguity is returned as
    None rather than resolved arbitrarily -- putting the same board on both
    sides of the split is the one failure this whole function exists to prevent.
    """
    want = frozenset(w.lower() for w in candidates)
    hits = [s for s, words in boards if want <= words]
    return hits[0] if len(hits) == 1 else None


def load_positions(db: Path, model: str, max_seed: int, collected: int = 0,
                   refresh_features: bool = False, decoys: Path | None = None):
    """Positions for training.

    `refresh_features` decides what a step-2+ row means. Off (the default and
    what every result so far used), features are computed once on the full
    candidate list and later steps reuse those rows -- so a step-2 row still
    says `n_candidates` = 25 when 24 remain, and every board-relative feature
    (`gaptop_*`, `peak_z`, `lead_margin`, `p_max_sigma2`, `cohesion`, the rank
    columns) describes a board that no longer exists. On, features are
    re-extracted for the words actually remaining at each step.

    Frozen is what Plackett-Luce assumes and what codenames/pl_reward.py needs
    to be exact. Measured, the assumption does not hold for this model:
    removing the top word changes its favourite among the rest 20% of the time
    and shifts logits by 0.52 on average. So the two options are a real choice
    -- consistency with the closed-form reward, or features that describe the
    actual board -- and this exists so it can be measured rather than assumed.
    """
    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    clue_index = {w.lower(): i for i, w in enumerate(stats.clue_words)}
    if WORD_STATS.exists():
        wstats = WordStats.load(WORD_STATS)
    else:
        print("building per-word column statistics (one-off, reads the full tensor)...", flush=True)
        wstats = WordStats.build(sims)
        wstats.save(WORD_STATS)
    swow = SwowTables.load(SWOW_TABLES) if SWOW_TABLES.exists() else None
    entity = EntitySims.load(ENTITY_SIMS) if ENTITY_SIMS.exists() else None
    # lm_pmi.npz has the same layout as entity_sims.npz, so it loads through
    # the same class rather than a near-duplicate one.
    pmi = EntitySims.load(LM_PMI) if LM_PMI.exists() else None
    extra = ExtraSims.load(EXTRA_SIMS) if EXTRA_SIMS.exists() else None
    norms = WordNorms.load(WORD_NORMS) if WORD_NORMS.exists() else None
    wordnet = ExtraSims.load(WORDNET_SIMS) if WORDNET_SIMS.exists() else None
    lexical = ExtraSims.load(LEXICAL_SIMS) if LEXICAL_SIMS.exists() else None
    print(f"SWOW: {'loaded' if swow else 'ABSENT'}   entity sims: "
          f"{'loaded' if entity else 'ABSENT'}   LM PMI: {'loaded' if pmi else 'ABSENT'}"
          f"   extra spaces: {'loaded' if extra else 'ABSENT'}"
          f"   norms: {'loaded' if norms else 'ABSENT'}"
          f"   wordnet: {'loaded' if wordnet else 'ABSENT'}"
          f"   lexical: {'loaded' if lexical else 'ABSENT'}")
    boards = board_lookup(max_seed, collected)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT clue, candidates, number, ranking FROM responses WHERE model=? AND number IS NOT NULL",
        (model,)).fetchall()
    conn.close()

    out, dropped = [], {"clue_oov": 0, "bad_ranking": 0, "no_seed": 0, "features": 0}
    for clue, cand_j, number, rank_j in rows:
        cand, rank = json.loads(cand_j), json.loads(rank_j)
        if sorted(w.lower() for w in cand) != sorted(w.lower() for w in rank):
            dropped["bad_ranking"] += 1
            continue
        seed = resolve_seed(rank, boards)
        if seed is None:
            dropped["no_seed"] += 1
            continue
        # Candidates are SHUFFLED before features are computed, and the
        # teacher's picks are tracked to their new rows. Feeding them in the
        # teacher's own order put the target at index 0 of every group, which
        # turned any positional artifact into the answer -- a NaN-filled rank
        # column leaked it once already (docs/log.md). With a shuffle, position
        # carries no information and the whole class of bug is dead.
        perm = random.Random(f"{seed}|{clue}|{number}").sample(range(len(rank)), len(rank))
        shuffled = [rank[i] for i in perm]
        where = {orig: new for new, orig in enumerate(perm)}
        targets = [where[j] for j in range(len(rank))]  # teacher's j-th pick -> its row
        feats = extract(clue, shuffled, number, sims, stats, wstats, clue_index, swow, entity, pmi, extra,
                        norms, wordnet, lexical)
        if feats is None:
            dropped["features"] += 1
            continue
        rec = {"seed": seed, "clue": clue, "k": int(number), "x": feats,
               "n": len(rank), "targets": targets}
        if refresh_features:
            # One extraction per step on the words still standing. Cheap --
            # it is the observed path, not a tree over possible ones.
            steps, taken, ok = [], [], True
            for j in range(min(int(number), len(rank) - 1)):
                keep = [r for r in range(len(rank)) if r not in taken]
                sub = [shuffled[r] for r in keep]
                f = extract(clue, sub, number, sims, stats, wstats, clue_index, swow, entity,
                            pmi, extra, norms, wordnet, lexical)
                if f is None:
                    ok = False
                    break
                steps.append((f, keep.index(targets[j])))
                taken.append(targets[j])
            if not ok:
                dropped["features"] += 1
                continue
            rec["steps"] = steps
        out.append(rec)
    if decoys is not None:
        got = decoy_positions(decoys, sims, stats, clue_index, wstats, swow, entity, pmi,
                              extra, norms, wordnet, lexical, dropped)
        print(f"decoy positions: {len(got)} from {decoys}")
        out.extend(got)
    return out, dropped


def decoy_positions(path: Path, sims, stats, clue_index, wstats, swow, entity, pmi,
                    extra, norms, wordnet, lexical, dropped: dict) -> list[dict]:
    """Positions from scripts/data/collect_decoy_data.py, in the same shape.

    Words drawn uniformly from the vocabulary are mixed into the candidate
    list. They fix the composition of the comparison set, which is what makes
    the absolute level identifiable: the board-normalised softmax is
    shift-invariant, so training on boards alone says which word is best and
    never whether any of them is good.

    Two differences from a DB position, both load-bearing:

    `k` is the TRUNCATION point, not the announced clue number -- rankings are
    cut at and including the first decoy, because below that line own-rate is
    33.2% against a 36% base rate, i.e. zero information. Setting k this way
    makes build_groups emit exactly the kept steps with no special case.

    `n_candidates` is overwritten with the board count. extract() computes it
    over whatever list it is given, so leaving it alone lets the decoy count
    enter as a feature -- the model could then read D directly, which both
    defeats the point and makes the invariance check vacuous.
    """
    ncand = FEATURE_NAMES.index("n_candidates")
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ranking"):
            dropped["decoy_short"] = dropped.get("decoy_short", 0) + 1
            continue
        cand = r["candidates"]
        perm = random.Random(f"{r['seed']}|{r['clue']}").sample(range(len(cand)), len(cand))
        shuffled = [cand[i] for i in perm]
        where = {w: i for i, w in enumerate(shuffled)}
        n_steps = min(r["cut"] + 1, len(shuffled) - 1)
        picks = [w for w in r["ranking"][:n_steps] if w in where]
        if n_steps < 1 or len(picks) < n_steps:
            dropped["decoy_short"] = dropped.get("decoy_short", 0) + 1
            continue
        feats = extract(r["clue"], shuffled, r["number"], sims, stats, wstats, clue_index,
                        swow, entity, pmi, extra, norms, wordnet, lexical)
        if feats is None:
            dropped["decoy_features"] = dropped.get("decoy_features", 0) + 1
            continue
        feats = np.array(feats, dtype=np.float64)
        feats[:, ncand] = r["n_board"]
        out.append({"seed": r["seed"], "clue": r["clue"], "k": n_steps, "x": feats,
                    "n": len(shuffled), "targets": [where[w] for w in picks],
                    "decoy": True, "n_decoys": r["n_decoys"],
                    "decoy_rows": [where[w] for w in r["decoys"] if w in where]})
    return out


def decoy_group_mask(positions: list[dict]) -> np.ndarray:
    """One flag per choice event, in build_groups order: is it a decoy group?

    Lets the two tasks be scored apart, so the headline R2 stays comparable
    with every run before decoys existed.
    """
    return np.array([bool(p.get("decoy")) for p in positions
                     for _ in range(min(p["k"], p["n"] - 1))], dtype=bool)


def build_groups(positions: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """Expand positions into Plackett-Luce choice events.

    Step j is a group over the words not yet picked, with the teacher's j-th
    pick as the target. Rows are in shuffled order and `targets` says where
    each pick landed, so the label is never at a fixed index -- see
    load_positions on why that matters.
    """
    X, y, groups, seeds = [], [], [], []
    for p in positions:
        if "steps" in p:                      # features re-extracted per step
            for rows, tgt in p["steps"]:
                lab = np.zeros(len(rows))
                lab[tgt] = 1.0
                X.append(rows); y.append(lab); groups.append(len(rows)); seeds.append(p["seed"])
            continue
        n, k = p["n"], p["k"]
        taken: list[int] = []
        for j in range(min(k, n - 1)):
            keep = [r for r in range(n) if r not in taken]
            rows = p["x"][keep]
            lab = np.zeros(len(rows))
            lab[keep.index(p["targets"][j])] = 1.0
            X.append(rows)
            y.append(lab)
            groups.append(len(rows))
            seeds.append(p["seed"])
            taken.append(p["targets"][j])
    return np.vstack(X), np.concatenate(y), groups, np.array(seeds)


STEP_DECAY = 0.75

# Per-event weight `STEP_DECAY ** step`. The PL expansion turns one teacher
# ranking into k choice events and weighted them equally, but steps 3 and 4
# score R2 0.11 and 0.07 -- the teacher's ranking tail is close to arbitrary
# once it has taken the words it wants -- while being 30% of the signal. Mean k
# in real games is 1.42, so they barely occur in play either.
#
# Measured over four schemes: 0.75**step gains +0.0102 nats at step 1 with
# pooled statistically unchanged. Training on step 1 ALONE is much worse
# (-0.0668 pooled) and does not even improve step 1, so the later steps are
# useful signal that simply must not dominate -- truncation is the wrong move,
# down-weighting is the right one.


def step_weights(positions: list[dict], decay: float = STEP_DECAY) -> np.ndarray:
    """One weight per choice event, decaying with depth into the turn.

    Decoy positions are exempt. The decay exists because the teacher's ranking
    tail is arbitrary once it has taken the words it wants; truncation at the
    first decoy already removes that tail, and the decoy term sits at the
    DEEPEST kept step -- so the decay would fall hardest on the one observation
    the position was collected for (mean cut 3.5 would weight it 0.75**3.5).
    """
    return np.array([1.0 if p.get("decoy") else decay ** j
                     for p in positions
                     for j in range(min(p["k"], p["n"] - 1))], dtype=np.float64)


def group_softmax_objective(groups: list[int], event_weights=None):
    """Conditional-logit loss: within each group, softmax cross-entropy.

    grad = p - y and hess = p(1-p), the standard multiclass softmax
    derivatives, applied per group rather than per row -- which is the whole
    difference between "score this word" and "choose among these words".

    Fully vectorised with `reduceat` rather than a Python loop over groups.
    The objective runs on every boosting round, so a loop over ~6k groups is
    ~9M interpreted iterations across a 1500-round fit and dominates the cost
    of actually building the trees.
    """
    sizes = np.asarray(groups, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    # Applied here rather than through LightGBM's `weight=`, so the scaling is
    # explicit and does not depend on whether a given version forwards dataset
    # weights into a custom objective's output.
    row_w = None if event_weights is None else np.repeat(event_weights, sizes)

    def obj(preds: np.ndarray, dset):
        y = dset.get_label()
        gmax = np.maximum.reduceat(preds, starts)
        e = np.exp(preds - np.repeat(gmax, sizes))
        gsum = np.add.reduceat(e, starts)
        p = e / np.repeat(gsum, sizes)
        grad, hess = p - y, np.maximum(p * (1.0 - p), 1e-6)
        if row_w is not None:
            grad, hess = grad * row_w, hess * row_w
        return grad, hess

    return obj


def top1_accuracy(preds: np.ndarray, groups: list[int], y: np.ndarray) -> float:
    return accuracy_on(preds, groups, y)


def baseline_accuracy(X: np.ndarray, groups: list[int], y: np.ndarray) -> float:
    """What plain numberbatch gets on the same groups -- the number to beat."""
    return top1_accuracy(X[:, FEATURE_NAMES.index("z_numberbatch")], groups, y)


def mcfadden_on(preds, groups, y, mask=None) -> float:
    """McFadden pseudo-R2: 1 - LL(model)/LL(uniform), per event.

    This is the stopping metric as well as the reported one, and those being
    the same thing is load-bearing. Early stopping used to watch tie-aware
    ACCURACY while the model was trained on group softmax and reported on R2 --
    three different quantities. Accuracy is a step function that plateaus and
    jitters, so the stopping point wandered: measured, otherwise-identical arms
    stopped anywhere between 138 and 652 trees, and the resulting noise was
    large enough to make three feature blocks that each help individually
    appear to hurt in combination. The null is computed per event as log(n)
    rather than assumed constant, because group sizes differ.
    """
    ll, null = group_log_loss(preds, groups, y)
    if mask is not None:
        ll, null = ll[mask], null[mask]
    return 1.0 - ll.mean() / null.mean()


def group_log_loss(preds, groups, y) -> tuple[np.ndarray, np.ndarray]:
    """Per choice event: the model's negative log-likelihood of the teacher's
    pick under a softmax over the group, and the uniform model's, log(n)."""
    bounds = np.concatenate([[0], np.cumsum(groups)])
    ll = np.empty(len(groups))
    null = np.empty(len(groups))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        s_ = preds[a:b] - preds[a:b].max()
        e = np.exp(s_)
        p = e / e.sum()
        ll[i] = -np.log(max(float(p[int(np.argmax(y[a:b]))]), 1e-12))
        null[i] = np.log(b - a)
    return ll, null


def train(Xtr, ytr, gtr, Xva, yva, gva, rounds: int, seed: int = 0,
          names: list[str] | None = None, event_weights=None):
    """LightGBM >= 4 takes a custom objective through `params["objective"]`.

    Early stopping is not optional here: this model will drive TRAINING accuracy
    to 0.99 while validation sits at the baseline (measured -- see docs/log.md).
    The stopping metric is McFadden R2, i.e. the training objective itself --
    see `mcfadden_on` for why watching accuracy instead was actively harmful.
    """
    import lightgbm as lgb

    names = list(names) if names is not None else list(FEATURE_NAMES)
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, feature_name=names, reference=dtr, free_raw_data=False)
    params = {
        "objective": group_softmax_objective(gtr, event_weights),
        # Swept late and mattered more than most feature blocks: 0.1/0.05/0.02/
        # 0.01/0.005/0.003 give R2 0.3505/0.3538/0.3559/0.3565/0.3570/0.3569
        # over three seeds. The curve flattens by 0.005, and 0.005 beats 0.01 by
        # +0.0005 -- inside the seed noise band -- while needing 2.1x the trees
        # (~4800 vs ~2300), which is inference cost the spymaster pays on every
        # clue. 0.01 takes +0.0027 of the available +0.0032 at half the price.
        "learning_rate": 0.01,
        # From scripts/tools/sweep_listener_params.py over 60 configs, ranked by
        # McFadden R2 on the calibration boards. The sweep's real finding is that
        # capacity barely matters here: the whole grid spans R2 0.3295-0.3329 and
        # the winner beat the old hand-set values by +0.0002 on held-out data,
        # which is noise. lambda_l2 is the only parameter that moved anything --
        # every config in the top ten had it at 10.0, the largest value tried, so
        # it sits AT the grid edge and the optimum may be higher. num_leaves is
        # near-irrelevant because early stopping trades tree size against tree
        # count (15 leaves x 1666 trees scored the same as 127 x 301); 127 is
        # kept because it reaches the plateau in the fewest trees, which is what
        # inference cost scales with.
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "seed": seed,
        "feature_pre_filter": False,
    }

    def feval(preds, _dset):
        return "r2", mcfadden_on(preds, gva, yva), True

    booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], feval=feval,
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    preds = booster.predict(Xva, raw_score=True)
    return booster, accuracy_on(preds, gva, yva)


def first_step_mask(positions: list[dict]) -> np.ndarray:
    """Which choice events are step 1 (pick the top word from the whole board).

    Reported separately because step 1 is the only one comparable with the
    "top-1 agreement" numbers measured elsewhere; steps 2+ choose from a
    smaller, harder set and drag the pooled figure down.
    """
    flags = []
    for p in positions:
        for j in range(min(p["k"], p["n"] - 1)):
            flags.append(j == 0)
    return np.array(flags)


def accuracy_on(preds, groups, y, mask=None) -> float:
    """Group top-1 accuracy, with ties scored as expectation, not as a win.

    The target sits at index 0 of every group -- features are built in the
    teacher's ranked order -- and `np.argmax` returns the FIRST maximum. So a
    model emitting constant scores would be graded 100% correct purely by
    tie-breaking, and the more strongly it is regularised the better it would
    look. Credit 1/(number tied at the top) instead, which is the expected
    accuracy under random tie-breaking and gives a constant model 1/n.
    """
    bounds = np.concatenate([[0], np.cumsum(groups)])
    hits = np.empty(len(groups))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        s_ = preds[a:b]
        top = s_.max()
        tied = int(np.sum(s_ >= top - 1e-12))
        target = int(np.argmax(y[a:b]))
        hits[i] = (1.0 / tied) if s_[target] >= top - 1e-12 else 0.0
    return float(hits.mean() if mask is None else hits[mask].mean())
