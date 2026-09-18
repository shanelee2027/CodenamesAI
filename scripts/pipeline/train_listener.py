"""Distil a local listener from cached LLM rankings.

Fits the conditional logit described in codenames/listener_features.py: a
LightGBM model scores every candidate word, a softmax over the board turns
those scores into a choice, and the teacher's observed pick is the response.
Each position contributes `k` choice events -- pick 1 from all n words, pick 2
from the remaining n-1, and so on to the announced clue number, which is
exactly as deep as the game ever reads a ranking (measured: 100% of turns read
at most 4 words, 58.5% read one).

**Split is by board seed, never by row.** One board yields many positions, one
position yields k groups, and one group yields n rows, so a random row split
leaks the answer three ways over. The seed is not stored on a cached response,
so it is recovered by matching the candidate set against regenerated boards --
positions that cannot be resolved are dropped rather than guessed, because a
mis-assigned seed silently contaminates the eval set.

Usage:
    python scripts/pipeline/train_listener.py
    python scripts/pipeline/train_listener.py --learning-curve
    python scripts/pipeline/train_listener.py --model deepinfra/openai/gpt-oss-120b+effort=low
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board
from codenames.clue_stats import ClueStats
from codenames.listener_features import (
    FEATURE_NAMES, N_FEATURES, EntitySims, SwowTables, WordStats, extract,
)
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

CACHE = PROJECT_ROOT / "cache"
DB = CACHE / "llm_store.db"
WORD_STATS = CACHE / "word_stats.npz"
SWOW_TABLES = CACHE / "swow.npz"
ENTITY_SIMS = CACHE / "entity_sims.npz"
DEFAULT_MODEL = "claude-sonnet-5+effort=medium"


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


def load_positions(db: Path, model: str, max_seed: int, collected: int = 0):
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
    print(f"SWOW: {'loaded' if swow else 'ABSENT'}   entity sims: {'loaded' if entity else 'ABSENT'}")
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
        feats = extract(clue, shuffled, number, sims, stats, wstats, clue_index, swow, entity)
        if feats is None:
            dropped["features"] += 1
            continue
        out.append({"seed": seed, "clue": clue, "k": int(number), "x": feats,
                    "n": len(rank), "targets": targets})
    return out, dropped


def build_groups(positions: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """Expand positions into Plackett-Luce choice events.

    Step j is a group over the words not yet picked, with the teacher's j-th
    pick as the target. Rows are in shuffled order and `targets` says where
    each pick landed, so the label is never at a fixed index -- see
    load_positions on why that matters.
    """
    X, y, groups, seeds = [], [], [], []
    for p in positions:
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


def group_softmax_objective(groups: list[int]):
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

    def obj(preds: np.ndarray, dset):
        y = dset.get_label()
        gmax = np.maximum.reduceat(preds, starts)
        e = np.exp(preds - np.repeat(gmax, sizes))
        gsum = np.add.reduceat(e, starts)
        p = e / np.repeat(gsum, sizes)
        return p - y, np.maximum(p * (1.0 - p), 1e-6)

    return obj


def top1_accuracy(preds: np.ndarray, groups: list[int], y: np.ndarray) -> float:
    return accuracy_on(preds, groups, y)


def baseline_accuracy(X: np.ndarray, groups: list[int], y: np.ndarray) -> float:
    """What plain numberbatch gets on the same groups -- the number to beat."""
    return top1_accuracy(X[:, FEATURE_NAMES.index("z_numberbatch")], groups, y)


def train(Xtr, ytr, gtr, Xva, yva, gva, rounds: int, seed: int = 0):
    """LightGBM >= 4 takes a custom objective through `params["objective"]`.

    Early stopping on held-out group accuracy is not optional here: with 21
    features and a few thousand choice events this model will drive TRAINING
    accuracy to 0.99 while validation sits at the baseline (measured -- see
    docs/log.md). The stopping metric is the tie-aware accuracy below, because
    the naive one rewards a degenerate constant model.
    """
    import lightgbm as lgb

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, feature_name=FEATURE_NAMES, reference=dtr, free_raw_data=False)
    params = {
        "objective": group_softmax_objective(gtr),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "feature_pre_filter": False,
    }

    def feval(preds, _dset):
        return "grp_acc", accuracy_on(preds, gva, yva), True

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="which teacher's cached rankings to distil")
    ap.add_argument("--max-seed", type=int, default=60, help="arena board seeds to try when resolving")
    ap.add_argument("--collected", type=int, default=10000,
                    help="generated-board seeds to try (scripts/data/collect_listener_data.py)")
    ap.add_argument("--val-frac", type=float, default=0.25, help="fraction of BOARD SEEDS held out")
    ap.add_argument("--rounds", type=int, default=3000, help="upper bound; early stopping decides")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--out", type=Path, default=CACHE / "listener_gbt.txt")
    args = ap.parse_args()

    t0 = time.time()
    positions, dropped = load_positions(args.db, args.model, args.max_seed, args.collected)
    print(f"teacher: {args.model}")
    print(f"usable positions: {len(positions)}   dropped: {dropped}")
    if not positions:
        raise SystemExit("no usable positions")

    seeds = sorted({p["seed"] for p in positions})
    rng = np.random.default_rng(args.seed)
    val_seeds = set(rng.choice(seeds, size=max(1, int(len(seeds) * args.val_frac)), replace=False).tolist())
    tr_pos = [p for p in positions if p["seed"] not in val_seeds]
    va_pos = [p for p in positions if p["seed"] in val_seeds]
    print(f"boards: {len(seeds)} total -> {len(seeds)-len(val_seeds)} train / {len(val_seeds)} val")
    print(f"positions: {len(tr_pos)} train / {len(va_pos)} val")

    Xtr, ytr, gtr, _ = build_groups(tr_pos)
    Xva, yva, gva, _ = build_groups(va_pos)
    print(f"choice events: {len(gtr)} train / {len(gva)} val   rows: {len(Xtr)} / {len(Xva)}")

    nb = Xva[:, FEATURE_NAMES.index("z_numberbatch")]
    step1 = first_step_mask(va_pos)
    base_all = accuracy_on(nb, gva, yva)
    base_s1 = accuracy_on(nb, gva, yva, step1)
    print(f"\nbaseline (numberbatch z alone): all steps {base_all:.4f}   step-1 only {base_s1:.4f}")
    base = base_all

    if args.learning_curve:
        print("\nlearning curve (train boards subsampled; val fixed):")
        tr_seeds = sorted({p["seed"] for p in tr_pos})
        for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
            keep = set(tr_seeds[: max(1, int(len(tr_seeds) * frac))])
            sub = [p for p in tr_pos if p["seed"] in keep]
            Xs, ys, gs, _ = build_groups(sub)
            _, acc = train(Xs, ys, gs, Xva, yva, gva, args.rounds, args.seed)
            print(f"  {frac:4.0%}  {len(sub):5d} positions, {len(gs):5d} events -> val top-1 {acc:.4f}"
                  f"   (baseline {base:.4f})")
        return

    booster, acc = train(Xtr, ytr, gtr, Xva, yva, gva, args.rounds, args.seed)
    preds = booster.predict(Xva, raw_score=True)
    m_all, m_s1 = accuracy_on(preds, gva, yva), accuracy_on(preds, gva, yva, step1)
    print(f"model    : all steps {m_all:.4f} ({m_all-base_all:+.4f})   "
          f"step-1 only {m_s1:.4f} ({m_s1-base_s1:+.4f})")

    imp = sorted(zip(FEATURE_NAMES, booster.feature_importance("gain")), key=lambda t: -t[1])
    print("\nfeature importance (gain):")
    for name, g in imp:
        print(f"  {name:22s} {g:12.1f}")
    booster.save_model(str(args.out))
    print(f"\nsaved -> {args.out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
