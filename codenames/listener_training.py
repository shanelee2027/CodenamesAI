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
DEFAULT_MODEL = "deepinfra/openai/gpt-oss-120b+effort=low"  # the teacher

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


class BoardIndex(list):
    """(seed, word set) pairs, plus an index from each word to the boards
    holding it, so a position resolves by intersecting a few sets instead of
    scanning every board. At 45,000 boards x 67,000 cached rankings the
    scan was billions of subset tests and dominated every load."""

    def __init__(self, boards: list[tuple[int, frozenset[str]]]):
        super().__init__(boards)
        self.by_word: dict[str, set[int]] = {}
        for i, (_, words) in enumerate(boards):
            for w in words:
                self.by_word.setdefault(w, set()).add(i)


def board_lookup(max_seed: int, collected: int = 0) -> BoardIndex:
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
    return BoardIndex(out)


def resolve_seed(candidates: list[str], boards: list[tuple[int, frozenset[str]]]) -> int | None:
    """Which board this position came from, or None if it is ambiguous.

    A position is a subset of its board, so the board is the one whose word set
    contains every candidate. Two boards sharing a 16-word subset is vanishingly
    unlikely (they are 25-word samples from 400), but ambiguity is returned as
    None rather than resolved arbitrarily -- putting the same board on both
    sides of the split is the one failure this whole function exists to prevent.
    """
    want = [w.lower() for w in candidates]
    if isinstance(boards, BoardIndex):
        sets = sorted((boards.by_word.get(w, set()) for w in set(want)), key=len)
        hits = set.intersection(*sets) if sets else set()
        return boards[next(iter(hits))][0] if len(hits) == 1 else None
    want_set = frozenset(want)
    hits = [s for s, words in boards if want_set <= words]
    return hits[0] if len(hits) == 1 else None


def _feature_cache_path() -> Path:
    """Where extracted features are cached, named by a fingerprint of
    everything they depend on: the feature list, this module and
    listener_features.py, and every side table's size and mtime. Change any
    of them and the name changes, so a stale cache is never read."""
    import hashlib

    h = hashlib.sha256(",".join(FEATURE_NAMES).encode())
    here = Path(__file__).resolve().parent
    for f in (here / "listener_features.py", here / "listener_training.py"):
        h.update(f.read_bytes())
    for f in (WORD_STATS, SWOW_TABLES, ENTITY_SIMS, LM_PMI, EXTRA_SIMS, WORD_NORMS, WORDNET_SIMS,
              LEXICAL_SIMS, CACHE / "similarity_tensor.npy", CACHE / "clue_stats.npz"):
        if f.exists():
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return CACHE / "training_data" / f"features_{h.hexdigest()[:12]}.pkl"


def load_soft_labels(path: Path, lm: str, temperature: float = 1.0, source: str = "own") -> dict:
    """Teacher distributions from scripts/data/collect_lm_distributions.py,
    keyed by (clue, candidates, number) -> one {word: probability} per step.

    `temperature` rescales the log-probabilities before renormalising: above 1
    softens an overconfident teacher, and 0 keeps only its argmax, which is the
    hard-label control -- same teacher, same positions, the distribution
    thrown away. `source` is what later steps were conditioned on
    (codenames/local_lm.py::DistributionStore): "own" for training.
    """
    from codenames.local_lm import DistributionStore

    out = {}
    for clue, cand, number, steps in DistributionStore(path).all(lm, source):
        path_words = scored_path(steps)
        dists = []
        for st, pick in zip(steps, path_words):
            lp = np.asarray(st["logprobs"], dtype=np.float64)
            if temperature == 0:
                # One-hot on the word the scorer actually took, not on every
                # word at the maximum: bf16 logits tie often enough (~0.25
                # resolution near 40) that "all maxima" split the label and
                # disagreed with the path the later steps were scored along.
                p = np.array([w == pick for w in st["remaining"]], dtype=np.float64)
            else:
                z = lp / temperature
                p = np.exp(z - np.logaddexp.reduce(z))
            dists.append(dict(zip(st["remaining"], p / p.sum())))
        out[(clue, tuple(cand), int(number))] = dists
    return out


def scored_path(steps: list[dict]) -> list[str]:
    """The word taken at each step of a stored position: the one missing from
    the next step's remaining words, and the argmax at the last step. Read off
    the data rather than recomputed, so a tie can never be broken differently
    from how the scorer broke it."""
    out = []
    for j, st in enumerate(steps):
        if j + 1 < len(steps):
            gone = set(st["remaining"]) - set(steps[j + 1]["remaining"])
            out.append(next(iter(gone)) if len(gone) == 1 else st["remaining"][int(np.argmax(st["logprobs"]))])
        else:
            out.append(st["remaining"][int(np.argmax(st["logprobs"]))])
    return out


def load_positions(db: Path, model: str, max_seed: int, collected: int = 0,
                   refresh_features: bool = False, decoys: Path | None = None,
                   soft_labels: dict | None = None, seed_filter=None):
    """Positions for training.

    `refresh_features` decides what a step-2+ row means. Off (the default and
    what every result so far used), features are computed once on the full
    candidate list and later steps reuse those rows -- so a step-2 row still
    says `n_candidates` = 25 when 24 remain, and every board-relative feature
    (`gaptop_*`, `peak_z`, `lead_margin`, `p_max_sigma2`, `cohesion`, the rank
    columns) describes a board that no longer exists. On, features are
    re-extracted for the words actually remaining at each step.

    `soft_labels` (from `load_soft_labels`) makes the local model the teacher:
    each step's label is its full distribution over the remaining words, and
    the words removed before step j are ITS argmax picks, not the ranking in
    the store -- so nothing from the API guesser enters but the position
    itself (board, clue, candidates), which our own sampler chose. Positions
    with no distribution are dropped. `seed_filter`, if given, keeps only positions
    whose board seed it accepts.

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

    import pickle

    fcache_path = _feature_cache_path()
    fcache = pickle.loads(fcache_path.read_bytes()) if fcache_path.exists() else {}
    n_cached = len(fcache)
    out, dropped = [], {"clue_oov": 0, "bad_ranking": 0, "no_seed": 0, "features": 0}
    if soft_labels is not None:
        dropped["no_soft"] = 0
    for clue, cand_j, number, rank_j in rows:
        cand, rank = json.loads(cand_j), json.loads(rank_j)
        if sorted(w.lower() for w in cand) != sorted(w.lower() for w in rank):
            dropped["bad_ranking"] += 1
            continue
        seed = resolve_seed(rank, boards)
        if seed is None:
            dropped["no_seed"] += 1
            continue
        if seed_filter is not None and not seed_filter(seed):
            continue
        soft = None
        if soft_labels is not None:
            soft = soft_labels.get((clue, tuple(cand), int(number)))
            if soft is None:
                dropped["no_soft"] += 1
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
        fkey = (clue, int(number), tuple(shuffled))
        if fkey in fcache:
            feats = fcache[fkey]
        else:
            feats = extract(clue, shuffled, number, sims, stats, wstats, clue_index, swow, entity, pmi, extra,
                            norms, wordnet, lexical)
            fcache[fkey] = feats
        if feats is None:
            dropped["features"] += 1
            continue
        rec = {"seed": seed, "clue": clue, "k": int(number), "x": feats,
               "n": len(rank), "targets": targets,
               "key": (clue, tuple(cand), int(number)), "words": shuffled}
        if soft is not None:
            # One label vector per step over ALL rows, in shuffled order, and
            # the local model's own picks as the targets that build_groups
            # removes: step j's distribution covers exactly the words its
            # argmax had not yet taken.
            rec["soft"] = [np.array([d.get(w, 0.0) for w in shuffled]) for d in soft]
            where_w = {w: r for r, w in enumerate(shuffled)}
            path_words = [next(iter(set(a) - set(b))) for a, b in zip(soft, soft[1:])]
            path_words.append(max(soft[-1], key=soft[-1].get))
            rec["targets"] = [where_w[w] for w in path_words]
            rec["k"] = len(soft)
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
    if len(fcache) > n_cached:
        fcache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = fcache_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(fcache, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(fcache_path)
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
            if "soft" in p and j < len(p["soft"]):
                lab = p["soft"][j][keep]
                lab = lab / lab.sum()
            else:
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


def split_positions(positions: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Train/val by BOARD seed, as train_listener.py has always drawn it.

    Board and decoy seeds are drawn SEPARATELY, so the board split is
    byte-identical whether or not decoys are mixed in. Pooling them would make
    every board accuracy incomparable with the run it is supposed to be
    measured against: adding seeds changes the draw for all of them. Here so
    an evaluation can rebuild the exact validation set a model was stopped on.
    """
    seeds = sorted({p["seed"] for p in positions if not p.get("decoy")})
    rng = np.random.default_rng(seed)
    val_seeds = set(rng.choice(seeds, size=max(1, int(len(seeds) * val_frac)), replace=False).tolist())
    dec_seeds = sorted({p["seed"] for p in positions if p.get("decoy")})
    if dec_seeds:
        rng_d = np.random.default_rng(seed + 1)
        val_seeds |= set(rng_d.choice(dec_seeds, size=max(1, int(len(dec_seeds) * val_frac)),
                                      replace=False).tolist())
    return ([p for p in positions if p["seed"] not in val_seeds],
            [p for p in positions if p["seed"] in val_seeds])


ASSOCIATIONS = CACHE / "associations.db"


def _assoc_norm(w: str) -> str:
    """Lowercase, accents and punctuation stripped: 'New York', 'new-york' and
    'newyork' are one word here, and so are the model's non-ASCII hyphens."""
    import unicodedata

    s = unicodedata.normalize("NFKD", w).lower()
    return "".join(ch for ch in s if ch.isalnum())


def load_associations(path: Path = ASSOCIATIONS) -> dict[str, list[set[str]]]:
    """clue (lowercase) -> one set of normalised words per sampled list,
    from scripts/data/collect_associations.py."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out: dict[str, list[set[str]]] = {}
    for clue, words in conn.execute("SELECT clue, words FROM lists"):
        out.setdefault(clue.lower(), []).append({_assoc_norm(w) for w in json.loads(words)})
    conn.close()
    return out


def association_counts(words: list[str], lists: list[set[str]]) -> np.ndarray:
    """How many of the lists name each word. A plural or singular form counts
    ('Bells' for Bell, 'bell' for Bells); each list counts at most once."""
    out = []
    for w in words:
        n = _assoc_norm(w)
        forms = {n, n + "s", n + "es"} | ({n[:-1]} if n.endswith("s") else set())
        out.append(sum(1 for lst in lists if forms & lst))
    return np.array(out, dtype=np.float64)


def association_targets(positions: list[dict], assoc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per row of build_groups(positions): the association count `y` and the
    number of lists `R` it is out of.

    Only step-1 rows carry one -- they hold every candidate exactly once, so
    each (clue, word) enters once per position rather than once per step it
    survives to. Every other row, and every position whose clue has no lists,
    gets R = 0, which removes it from the Poisson term exactly (its mean is
    R * rate = 0 and so is its target)."""
    ys, rs = [], []
    for p in positions:
        if "steps" in p:
            raise ValueError("association targets need full-board rows; drop --refresh-features")
        lists = None if p.get("decoy") else assoc.get(p["clue"].lower())
        for j in range(min(p["k"], p["n"] - 1)):
            m = p["n"] - j
            if j == 0 and lists:
                ys.append(association_counts(p["words"], lists))
                rs.append(np.full(m, float(len(lists))))
            else:
                ys.append(np.zeros(m))
                rs.append(np.zeros(m))
    return np.concatenate(ys), np.concatenate(rs)


def fit_rate_link(s: np.ndarray, y: np.ndarray, r: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Poisson GLM y ~ Poisson(r * exp(a*s + b)) in (a, b), by Newton.

    Used two ways: to pick the slope a from an existing listener before
    training (a listener's scores are choice log-odds; how many association
    nats one of those is worth is not 1 by assumption), and to give every
    model its best (a, b) on training rows before it is scored on validation
    ones, so a listener never trained on associations is not penalised for an
    arbitrary level or scale."""
    m = r > 0
    s, y, r = s[m], y[m], r[m]
    a, b = 1.0, float(np.log(max(y.sum(), 1e-9) / r.sum()) - s.mean())
    for _ in range(iters):
        mu = r * np.exp(np.clip(a * s + b, -30, 10))
        g = np.array([np.dot(mu - y, s), np.sum(mu - y)])
        h = np.array([[np.dot(mu, s * s), np.dot(mu, s)], [np.dot(mu, s), mu.sum()]])
        step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Total Poisson deviance, 2 * sum(y log(y/mu) - (y - mu)): the generalised
    KL divergence between counts and means, i.e. the Bregman divergence of
    x log x -- the loss the association term minimises."""
    mu = np.maximum(mu, 1e-12)
    t = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return float(2.0 * np.sum(t - (y - mu)))


def association_report(s_tr, y_tr, r_tr, s_va, y_va, r_va, groups_va, positions_va) -> dict:
    """How well a listener's scores predict held-out association counts.

    (a, b) is refitted on training rows for every model -- see fit_rate_link.
    `d2` is the share of Poisson deviance explained against one constant rate
    for every row. `level_rho` is the one number the board softmax cannot
    learn: per position, the predicted total association mass of its board,
    sum_w r*exp(a*s_w + b), against the observed total, Spearman over
    positions. `within_rho` is the complementary, board-relative part: each
    row's predicted and observed count with its position's mean removed from
    both, which is what the softmax already sees."""
    from scipy.stats import spearmanr

    a, b = fit_rate_link(s_tr, y_tr, r_tr)
    m = r_va > 0
    mu = r_va * np.exp(np.clip(a * s_va + b, -30, 10))
    null = r_va[m] * (y_va[m].sum() / r_va[m].sum())
    d2 = 1.0 - poisson_deviance(y_va[m], mu[m]) / poisson_deviance(y_va[m], null)

    starts = np.concatenate([[0], np.cumsum(groups_va)[:-1]])
    pred_tot, obs_tot, pw, ow = [], [], [], []
    for st, g in zip(starts, groups_va):
        sl = slice(st, st + g)
        if r_va[st] == 0 or g < 2:
            continue
        pred_tot.append(mu[sl].sum())
        obs_tot.append(y_va[sl].sum())
        pw.extend(np.log(mu[sl]) - np.log(mu[sl]).mean())
        ow.extend(y_va[sl] - y_va[sl].mean())
    return {"a": a, "b": b, "d2": d2, "rows": int(m.sum()), "boards": len(pred_tot),
            "level_rho": float(spearmanr(pred_tot, obs_tot)[0]),
            "within_rho": float(spearmanr(pw, ow)[0])}


def group_softmax_objective(groups: list[int], event_weights=None, assoc=None):
    """Conditional-logit loss: within each group, softmax cross-entropy.

    grad = p - y and hess = p(1-p), the standard multiclass softmax
    derivatives, applied per group rather than per row -- which is the whole
    difference between "score this word" and "choose among these words".

    `assoc = (y, r, weight, slope, offset)` adds a Poisson term on association
    counts (association_targets): row count y ~ Poisson(r * exp(slope*s +
    offset)), contributing weight * (mu - y) * slope to the gradient and
    weight * mu * slope^2 to the hessian. That term is not invariant to adding
    a constant to a group, so it is what pins the per-clue level the softmax
    leaves free -- see docs/log.md, "association counts".

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
        if assoc is not None:
            ay, ar, lam, slope, offset = assoc
            mu = ar * np.exp(np.clip(slope * preds + offset, -30, 10))
            grad = grad + lam * slope * (mu - ay)
            hess = hess + lam * slope * slope * mu
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
    """Per choice event: the model's cross-entropy against the label under a
    softmax over the group, and the uniform model's, log(n).

    With a one-hot label that is the negative log-likelihood of the teacher's
    pick; with a soft label (a local model's distribution) it is the expected
    one under that distribution, which is the quantity soft-label training
    minimises."""
    bounds = np.concatenate([[0], np.cumsum(groups)])
    ll = np.empty(len(groups))
    null = np.empty(len(groups))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        s_ = preds[a:b] - preds[a:b].max()
        logp = s_ - np.log(np.exp(s_).sum())
        ll[i] = -float(np.dot(y[a:b], np.maximum(logp, np.log(1e-12))))
        null[i] = np.log(b - a)
    return ll, null


def train(Xtr, ytr, gtr, Xva, yva, gva, rounds: int, seed: int = 0,
          names: list[str] | None = None, event_weights=None, assoc=None):
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
        "objective": group_softmax_objective(gtr, event_weights, assoc),
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
