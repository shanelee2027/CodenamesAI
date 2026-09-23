"""Which listener better predicts real guessers' picks? Free: no API calls.

Scores distilled listeners (LightGBM boosters) and, as a reference, the local
language model itself, on two held-out sets of OBSERVED picks:

  gpt-oss holdout   gpt-oss rankings on collected boards no listener trained on
                    (seeds 1e6+40000 and up) -- the incumbent teacher's home turf
  Sonnet            Sonnet's rankings from arena games (seeds < 60), which no
                    listener here trained on -- a guesser neither teacher is,
                    and the one the frozen eval suite plays against

Every model is scored on exactly the same choice events: positions the local
model has no distribution for are dropped from all of them, not just from its
row. The metric is McFadden R^2 of the observed pick, 1 - LL(model)/LL(uniform),
pooled and at step 1 (the guesser's first pick, 65% of what play reads).

The local model's distributions need a temperature: raw, it is far more
confident than any guesser behaves. It is fitted on gpt-oss TRAINING positions
(maximum likelihood of gpt-oss's picks), never on an evaluation set.

    python scripts/tools/compare_listeners.py \\
        --booster incumbent=cache/listener_gbt_oss_recipe.txt \\
        --booster qwen=cache/listener_gbt_qwen.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from codenames.listener_training import (
    DB, DEFAULT_MODEL, build_groups, first_step_mask, group_log_loss, load_positions, load_soft_labels,
)
from codenames.local_lm import DEFAULT_DIST_DB, DEFAULT_LM

SONNET = "claude-sonnet-5+effort=medium"
HOLDOUT_START = 1_040_000


def observed_logloss(positions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per choice event: -log q(observed pick) under the positions' soft labels
    (the local model's distribution), and log(n)."""
    ll, null = [], []
    for p in positions:
        taken: list[int] = []
        for j in range(min(p["k"], p["n"] - 1)):
            keep = [r for r in range(p["n"]) if r not in taken]
            q = p["soft"][j][keep]
            q = q / q.sum()
            ll.append(-np.log(max(float(q[keep.index(p["targets"][j])]), 1e-12)))
            null.append(np.log(len(keep)))
            taken.append(p["targets"][j])
    return np.array(ll), np.array(null)


def fit_temperature(train_positions: list[dict]) -> float:
    """Maximum-likelihood temperature for the local model against the teacher's
    observed picks, on training positions only."""
    best = (np.inf, 1.0)
    for t in (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12):
        for p in train_positions:
            p["soft"] = tempered(p, t)
        ll, _ = observed_logloss(train_positions)
        best = min(best, (ll.mean(), t))
    return best[1]


def tempered(p: dict, t: float) -> list[np.ndarray]:
    """The position's T=1 distributions raised to 1/t (renormalised later,
    over whichever words remain)."""
    return [np.exp(np.log(np.maximum(v, 1e-300)) / t) for v in p["_raw"]]


def attach_raw(positions: list[dict]) -> None:
    for p in positions:
        p["_raw"] = [v.copy() for v in p["soft"]]


def r2(ll: np.ndarray, null: np.ndarray, mask=None) -> float:
    if mask is not None:
        ll, null = ll[mask], null[mask]
    return 1.0 - ll.mean() / null.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--dists", type=Path, default=DEFAULT_DIST_DB)
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--fit-positions", type=int, default=3000,
                    help="gpt-oss training positions used to fit the temperature")
    args = ap.parse_args()

    import lightgbm as lgb

    raw = load_soft_labels(args.dists, args.lm, 1.0)
    fit = load_positions(DB, DEFAULT_MODEL, 0, 40000, soft_labels=raw)[0][:args.fit_positions]
    attach_raw(fit)
    t_star = fit_temperature(fit)
    print(f"temperature fitted on {len(fit)} gpt-oss training positions: T* = {t_star:g}\n")

    sets = {
        "gpt-oss holdout": load_positions(DB, DEFAULT_MODEL, 0, 45000, soft_labels=raw,
                                          seed_filter=lambda s: s >= HOLDOUT_START)[0],
        "Sonnet": load_positions(DB, SONNET, 60, 0, soft_labels=raw)[0],
    }
    boosters = {}
    for spec in args.booster:
        name, _, path = spec.partition("=")
        boosters[name] = lgb.Booster(model_file=path)

    for set_name, pos in sets.items():
        if not pos:
            print(f"{set_name}: no positions with a local-model distribution yet\n")
            continue
        attach_raw(pos)
        X, y, groups, _ = build_groups([{k: v for k, v in p.items() if k != "soft"} for p in pos])
        s1 = first_step_mask(pos)
        print(f"== {set_name}: {len(pos)} positions, {len(groups)} choice events")
        print(f"   {'model':28s} {'R2 pooled':>10s} {'R2 step 1':>10s} {'R2 steps 2+':>12s}")
        for name, b in boosters.items():
            ll, null = group_log_loss(b.predict(X, raw_score=True), groups, y)
            print(f"   {name:28s} {r2(ll, null):10.4f} {r2(ll, null, s1):10.4f} {r2(ll, null, ~s1):12.4f}")
        lm_name = args.lm.split("/")[-1]
        for t in (1.0, t_star):
            for p in pos:
                p["soft"] = tempered(p, t)
            ll, null = observed_logloss(pos)
            label = f"{lm_name} raw, T={t:g}"
            print(f"   {label:28s} {r2(ll, null):10.4f} {r2(ll, null, s1):10.4f} {r2(ll, null, ~s1):12.4f}")
        print()


if __name__ == "__main__":
    main()
