"""Score the listener once, on boards nothing has ever been selected on.

**Why re-splitting the existing data would not do.** The validation set has had
roughly twenty-five decisions made against it this session -- every feature
block, the prune, the hyperparameters, the learning rate, the step weighting.
Its R2 is therefore an optimistic estimate of generalisation by an unknown
amount, and carving a "test" set out of the same boards does not fix that,
because the feature set and hyperparameters were chosen against data those
boards overlap. The only clean answer is boards the project has never seen.

So the test positions were collected afterwards, from a board-seed range beyond
everything used for training or selection (`collect_listener_data.py --start`),
and every modelling decision was frozen before they were bought.

Trains on all non-test positions -- the old train and val together, since the
decisions those informed are already made -- and scores the result once.

Usage:
    python scripts/tools/eval_listener_holdout.py --test-start 40000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import codenames.listener_training as T
from codenames.listener_features import FEATURE_NAMES  # noqa: E402

SEED_BASE = 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--model", default="deepinfra/openai/gpt-oss-120b+effort=low")
    ap.add_argument("--collected", type=int, default=45000)
    ap.add_argument("--test-start", type=int, default=40000,
                    help="board-seed offset at which the untouched test range begins")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    positions, dropped = T.load_positions(args.db, args.model, 60, args.collected)
    cut = SEED_BASE + args.test_start
    train = [p for p in positions if p["seed"] < cut]
    test = [p for p in positions if p["seed"] >= cut]
    print(f"usable positions {len(positions)}  dropped {dropped}")
    print(f"train (old train+val) {len(train)}   TEST (never seen) {len(test)}")
    if not test:
        raise SystemExit("no test positions -- has the test range been collected?")

    Xtr, ytr, gtr, _ = T.build_groups(train)
    Xte, yte, gte, _ = T.build_groups(test)
    w = T.step_weights(train)
    step = np.array([j for p in test for j in range(min(p["k"], p["n"] - 1))])
    b = np.concatenate([[0], np.cumsum(gte)])
    B = list(zip(b[:-1], b[1:]))
    tgt = np.array([int(np.argmax(yte[a:c])) for a, c in B])
    null = np.log(np.array(gte, dtype=float))
    s1 = step == 0
    print(f"choice events: {len(gtr)} train / {len(gte)} test ({int(s1.sum())} at step 1)", flush=True)

    def nll_of(raw):
        o = np.empty(len(B))
        for i, (a, c) in enumerate(B):
            sc = raw[a:c] - raw[a:c].max()
            e = np.exp(sc)
            p = e / e.sum()
            o[i] = -np.log(max(float(p[tgt[i]]), 1e-12))
        return o

    # Baselines on the same events, so the test number has a floor and a ceiling
    # to sit between rather than standing alone.
    from codenames.listener_features import p_is_max
    zi = FEATURE_NAMES.index("z_numberbatch")
    gauss = np.empty(len(B))
    for i, (a, c) in enumerate(B):
        p = p_is_max(np.asarray(Xte[a:c, zi], dtype=np.float64), 2.4)
        t = p.sum()
        p = p / t if t > 0 else np.full(len(p), 1.0 / len(p))
        gauss[i] = -np.log(max(float(p[tgt[i]]), 1e-12))

    lls = []
    for sd in range(args.seeds):
        bst, _ = T.train(Xtr, ytr, gtr, Xte, yte, gte, 8000, sd, list(FEATURE_NAMES), w)
        lls.append(nll_of(bst.predict(Xte, raw_score=True)))
        print(f"  seed {sd}: trees={bst.best_iteration}", flush=True)

    def r2(ll, m=None):
        return 1 - (ll.mean() / null.mean() if m is None else ll[m].mean() / null[m].mean())

    print(f"\n{'model':22s} {'test R2':>9s} {'step-1':>9s}")
    print(f"{'uniform':22s} {0.0:9.4f} {0.0:9.4f}")
    print(f"{'Gaussian p_is_max 2.4':22s} {r2(gauss):9.4f} {r2(gauss, s1):9.4f}")
    v = [r2(l) for l in lls]
    v1 = [r2(l, s1) for l in lls]
    print(f"{'distilled GBT':22s} {np.mean(v):9.4f} {np.mean(v1):9.4f}"
          f"   (seeds {', '.join(f'{x:.4f}' for x in v)})")

    d = gauss - np.mean(lls, axis=0)
    idx = np.random.default_rng(1).integers(0, len(d), size=(4000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    print(f"\nGaussian -> GBT on untouched boards: {d.mean():+.4f} nats [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
