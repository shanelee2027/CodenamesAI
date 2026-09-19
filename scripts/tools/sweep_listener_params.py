"""Hyperparameter sweep for the distilled listener, ranked by McFadden's R².

The params in train_listener.py were hand-set and never tuned; only the number
of trees was ever chosen by the data (early stopping). This sweeps the ones
that control capacity.

**Why McFadden's pseudo-R² rather than a plain R².** There is no continuous
response here -- the model is a conditional logit over board words, so the
ordinary R² has nothing to be the variance of. McFadden's is the standard
analogue for exactly this model class:

    R2 = 1 - LL(model) / LL(null)

with the null being the uniform choice over each candidate set, whose group
sizes differ, so LL(null) is computed per event as log(n) rather than assumed
constant. It is a monotone transform of mean validation log loss, so ranking on
it agrees with the proper scoring rule the model is trained under.

**Selection is on the calibration boards, not the validation boards.** Early
stopping already consumes whatever set it watches, so choosing hyperparameters
on that same set would pick the config that best overfits the stopping signal
and the reported number would be optimistic. Fit on `fit`, early-stop and rank
on `calib`, then re-score only the winner on `val`, which nothing has touched.

Parallel over configs rather than within LightGBM: the dataset is built once in
the parent and inherited copy-on-write by forked workers, and each worker gets
`nproc / workers` threads so the box is not oversubscribed.

Usage:
    python scripts/tools/sweep_listener_params.py --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.argv = [sys.argv[0]]
import scripts.pipeline.train_listener as T  # noqa: E402
from codenames.listener_features import FEATURE_NAMES  # noqa: E402

# Second pass. The first sweep ran at 32 features and put every one of its top
# ten configs at lambda_l2=10.0, the largest value it tried -- i.e. the optimum
# was at or beyond the grid edge and unresolved. This extends l2 upward and
# re-centres the other two on where the first pass landed, now at 44 features
# and with step weighting on.
GRID = {
    "num_leaves": [31, 63, 127, 255],
    "min_data_in_leaf": [50, 100, 250],
    "lambda_l2": [10.0, 30.0, 100.0, 300.0],
}

_D: dict = {}  # filled in the parent, inherited by forked workers


def mcfadden(preds: np.ndarray, groups: list[int], y: np.ndarray) -> tuple[float, float]:
    """(pseudo-R², mean log loss) for group-softmax scores."""
    b = np.concatenate([[0], np.cumsum(groups)])
    ll = np.empty(len(groups))
    null = np.empty(len(groups))
    for i, (lo, hi) in enumerate(zip(b[:-1], b[1:])):
        s = preds[lo:hi] - preds[lo:hi].max()
        e = np.exp(s)
        p = e / e.sum()
        ll[i] = -np.log(max(float(p[int(np.argmax(y[lo:hi]))]), 1e-12))
        null[i] = np.log(hi - lo)
    return 1.0 - ll.mean() / null.mean(), float(ll.mean())


def run_one(cfg: dict) -> dict:
    import lightgbm as lgb

    Xf, yf, gf, Xc, yc, gc = _D["fit"] + _D["cal"]
    params = {
        "objective": T.group_softmax_objective(gf, _D["wf"]),
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 0,
        "feature_pre_filter": False,
        "num_threads": _D["threads"],
        **cfg,
    }
    dtr = lgb.Dataset(Xf, label=yf, feature_name=list(FEATURE_NAMES), free_raw_data=False)
    dva = lgb.Dataset(Xc, label=yc, feature_name=list(FEATURE_NAMES), reference=dtr, free_raw_data=False)

    def feval(preds, _d):
        return "r2", mcfadden(preds, gc, yc)[0], True

    t0 = time.time()
    bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], feval=feval,
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    r2, ll = mcfadden(bst.predict(Xc, raw_score=True), gc, yc)
    return {**cfg, "trees": bst.best_iteration, "r2": r2, "logloss": ll, "secs": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--model", default="deepinfra/openai/gpt-oss-120b+effort=low")
    ap.add_argument("--collected", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    positions, _ = T.load_positions(args.db, args.model, 60, args.collected)
    seeds = sorted({p["seed"] for p in positions})
    rng = np.random.default_rng(args.seed)
    val = set(rng.choice(seeds, size=int(len(seeds) * 0.25), replace=False).tolist())
    tr_pos = [p for p in positions if p["seed"] not in val]
    va_pos = [p for p in positions if p["seed"] in val]
    tr_seeds = sorted({p["seed"] for p in tr_pos})
    cal = set(np.random.default_rng(args.seed + 7)
              .choice(tr_seeds, size=int(len(tr_seeds) * 0.2), replace=False).tolist())

    _D["fit"] = list(T.build_groups([p for p in tr_pos if p["seed"] not in cal])[:3])
    _D["cal"] = list(T.build_groups([p for p in tr_pos if p["seed"] in cal])[:3])
    _D["val"] = list(T.build_groups(va_pos)[:3])
    _D["wf"] = T.step_weights([p for p in tr_pos if p["seed"] not in cal])
    _D["threads"] = max(1, (os.cpu_count() or 8) // args.workers)
    print(f"events: {len(_D['fit'][2])} fit / {len(_D['cal'][2])} calib / {len(_D['val'][2])} val")

    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(configs)} configs, {args.workers} workers x {_D['threads']} threads", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(run_one, configs), 1):
            results.append(r)
            print(f"  [{i:3d}/{len(configs)}] leaves={r['num_leaves']:4d} "
                  f"min_leaf={r['min_data_in_leaf']:4d} l2={r['lambda_l2']:5.1f}  "
                  f"trees={r['trees']:4d}  R2={r['r2']:.4f}  ({r['secs']:.0f}s)", flush=True)
    print(f"\nswept in {(time.time()-t0)/60:.1f} min")

    results.sort(key=lambda r: -r["r2"])
    print(f"\ntop 10 by calibration-set McFadden R2:")
    print(f"  {'leaves':>7s} {'min_leaf':>9s} {'l2':>6s} {'trees':>6s} {'R2':>8s} {'logloss':>9s}")
    for r in results[:10]:
        print(f"  {r['num_leaves']:7d} {r['min_data_in_leaf']:9d} {r['lambda_l2']:6.1f} "
              f"{r['trees']:6d} {r['r2']:8.4f} {r['logloss']:9.4f}")

    CURRENT = {"num_leaves": 127, "min_data_in_leaf": 100, "lambda_l2": 10.0}
    base = next(r for r in results if all(r[k] == v for k, v in CURRENT.items()))
    print(f"\ncurrent config {CURRENT} -> R2 {base['r2']:.4f} "
          f"(rank {results.index(base)+1} of {len(results)})")

    # Only the winner is scored on val, which nothing has touched.
    import lightgbm as lgb
    best = {k: results[0][k] for k in GRID}
    Xf, yf, gf, Xc, yc, gc = _D["fit"] + _D["cal"]
    Xv, yv, gv = _D["val"]
    for label, cfg in (("current", CURRENT), ("swept", best)):
        params = {"objective": T.group_softmax_objective(gf, _D["wf"]), "learning_rate": 0.05,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "verbosity": -1, "seed": 0, "feature_pre_filter": False, **cfg}
        dtr = lgb.Dataset(Xf, label=yf, feature_name=list(FEATURE_NAMES), free_raw_data=False)
        dva = lgb.Dataset(Xc, label=yc, feature_name=list(FEATURE_NAMES), reference=dtr, free_raw_data=False)
        bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        feval=lambda p, d: ("r2", mcfadden(p, gc, yc)[0], True),
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        r2, ll = mcfadden(bst.predict(Xv, raw_score=True), gv, yv)
        print(f"  held-out val, {label:8s} {cfg} -> R2 {r2:.4f}  logloss {ll:.4f}")


if __name__ == "__main__":
    main()
