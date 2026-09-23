"""Which of the 55 features actually earn their place?

**Leave-one-out, not SHAP.** SHAP says how much the fitted model *uses* a
feature; it cannot say whether the model would do just as well without it,
because a feature whose information is recoverable from its correlates can be
heavily used and still free to drop. With five blocks of association and
similarity features that overlap by construction, that distinction is the whole
question. So each feature is scored by retraining without it.

**And a joint check afterwards.** Leave-one-out systematically understates
redundant groups: if two features carry the same information, dropping either
alone costs nothing, and the naive conclusion is that both are droppable. The
second pass therefore drops the whole LOO-null set at once and measures that,
which is the number that actually matters.

**Selection on the calibration boards.** Early stopping already consumes
whatever set it watches, so choosing features on that same set would pick
whatever best overfits the stopping signal. Fit on `fit`, early-stop and rank on
`calib`, report the final comparison on `val`, which nothing has touched.

Usage:
    python scripts/tools/prune_listener_features.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import codenames.listener_training as T
from codenames.listener_features import FEATURE_NAMES  # noqa: E402

_D: dict = {}


def fit_score(keep: list[str], on: str = "cal") -> tuple[float, int]:
    import lightgbm as lgb

    Xf, yf, gf = _D["fit"]
    Xc, yc, gc = _D["cal"]
    cols = [FEATURE_NAMES.index(n) for n in keep]
    params = {
        "objective": T.group_softmax_objective(gf), "learning_rate": 0.05,
        "num_leaves": 127, "min_data_in_leaf": 100, "lambda_l2": 10.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "verbosity": -1, "seed": 0, "feature_pre_filter": False,
        "num_threads": _D["threads"],
    }
    dtr = lgb.Dataset(Xf[:, cols], label=yf, feature_name=keep, free_raw_data=False)
    dva = lgb.Dataset(Xc[:, cols], label=yc, feature_name=keep, reference=dtr, free_raw_data=False)
    bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                    feval=lambda p, d: ("r2", T.mcfadden_on(p, gc, yc), True),
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    X, y, g = _D[on]
    return T.mcfadden_on(bst.predict(X[:, cols], raw_score=True), g, y), bst.best_iteration


FULL = "__full__"


def loo(name: str) -> tuple[str, float]:
    """One job. `FULL` means "fit with everything", which is the baseline.

    **The baseline is a worker job, not a parent one, and that is load-bearing.**
    Training in the parent before `ProcessPoolExecutor` forks starts LightGBM's
    OpenMP thread pool; the children then inherit a mutex held by a thread that
    does not exist in them and deadlock on their first LightGBM call -- eight
    workers at zero CPU, forever, which is exactly what happened the first time
    this was run. Nothing may touch LightGBM in the parent until the pool is
    closed.
    """
    keep = list(FEATURE_NAMES) if name == FULL else [n for n in FEATURE_NAMES if n != name]
    return name, fit_score(keep)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--model", default="deepinfra/openai/gpt-oss-120b+effort=low")
    ap.add_argument("--collected", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="drop features whose leave-one-out delta is at or below this")
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
    _D["threads"] = max(1, (os.cpu_count() or 8) // args.workers)
    print(f"events: {len(_D['fit'][2])} fit / {len(_D['cal'][2])} calib / {len(_D['val'][2])} val")

    t0 = time.time()
    jobs = [FULL] + list(FEATURE_NAMES)
    raw = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, r2) in enumerate(ex.map(loo, jobs), 1):
            raw[name] = r2
            print(f"  [{i:2d}/{len(jobs)}] {name:18s} R2 {r2:.5f}", flush=True)
    base = raw.pop(FULL)
    results = {n: base - r2 for n, r2 in raw.items()}
    print(f"\nfull {len(FEATURE_NAMES)} features: calib R2 {base:.4f}")
    print(f"leave-one-out done in {(time.time()-t0)/60:.1f} min")

    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    print(f"\n{'feature':18s} {'LOO value':>10s}")
    for n, v in ranked:
        print(f"  {n:18s} {v:+10.5f}")

    drop = [n for n, v in ranked if v <= args.threshold]
    keep = [n for n in FEATURE_NAMES if n not in drop]
    print(f"\n{len(drop)} features at or below {args.threshold:+.5f}: {', '.join(drop)}")

    # The joint check. Individually-null features can be jointly load-bearing.
    joint, jt = fit_score(keep)
    print(f"\ncalib: full {base:.4f} ({len(FEATURE_NAMES)} feat) -> pruned {joint:.4f} "
          f"({len(keep)} feat, {jt} trees)   delta {joint-base:+.5f}")
    vf, _ = fit_score(list(FEATURE_NAMES), on="val")
    vp, _ = fit_score(keep, on="val")
    print(f"val  : full {vf:.4f} -> pruned {vp:.4f}   delta {vp-vf:+.5f}")
    print("\nkeep = [" + ", ".join(f'"{n}"' for n in keep) + "]")


if __name__ == "__main__":
    main()
