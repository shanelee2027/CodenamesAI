"""Which listener better predicts real guessers' picks? Free: no API calls.

Scores distilled listeners (LightGBM boosters) and, as a reference, the local
language model itself, on two held-out sets of OBSERVED picks:

  Sonnet            Sonnet's rankings from arena games (seeds < 60), which no
                    listener here trained on -- the guesser the frozen eval
                    suite plays against, and the primary comparison
  gpt-oss holdout   gpt-oss rankings on collected boards no listener trained on
                    (seeds 1e6+40000 and up) -- the incumbent teacher's home turf

Targets are always the guesser's actual picks, never the local model's. Every
model is scored on exactly the same choice events: on the Sonnet set,
positions the local model has no distribution for are dropped from every row.

The local model is scored as it is, at T=1: fitting its temperature to some
guesser's picks would make that guesser part of the teacher. On the Sonnet set
its step-j distribution conditions on Sonnet's own first j-1 picks (source
"teacher" in collect_lm_distributions.py), the same conditioning every booster
gets. On the gpt-oss holdout only own-path distributions exist, which match the
observed path at step 1 alone, so it is reported there at step 1 only.

The metric is McFadden R^2 of the observed pick, 1 - LL(model)/LL(uniform),
pooled and at step 1 (the guesser's first pick, 65% of what play reads).

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


def lm_logloss(positions: list[dict], dists: dict, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per choice event: -log q(observed pick) under the local model's
    distribution for that step, and log(n). `steps` caps the steps scored."""
    ll, null = [], []
    for p in positions:
        d = dists[p["key"]]
        taken: list[int] = []
        for j in range(min(p["k"], p["n"] - 1, steps or p["n"])):
            by_word = {w.lower(): v for w, v in d[j].items()}
            keep = [r for r in range(p["n"]) if r not in taken]
            q = np.array([by_word.get(p["words"][r].lower(), 0.0) for r in keep])
            q = q / q.sum()
            ll.append(-np.log(max(float(q[keep.index(p["targets"][j])]), 1e-12)))
            null.append(np.log(len(keep)))
            taken.append(p["targets"][j])
    return np.array(ll), np.array(null)


def r2(ll: np.ndarray, null: np.ndarray, mask=None) -> float:
    if mask is not None:
        ll, null = ll[mask], null[mask]
    return 1.0 - ll.mean() / null.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--dists", type=Path, default=DEFAULT_DIST_DB)
    ap.add_argument("--lm", default=DEFAULT_LM)
    args = ap.parse_args()

    import lightgbm as lgb

    boosters = {}
    for spec in args.booster:
        name, _, path = spec.partition("=")
        boosters[name] = lgb.Booster(model_file=path)
    lm_name = args.lm.split("/")[-1]

    sonnet_dists = load_soft_labels(args.dists, args.lm, 1.0, source=SONNET)
    own_dists = load_soft_labels(args.dists, args.lm, 1.0, source="own")
    sets = {
        "Sonnet": (load_positions(DB, SONNET, 60, 0)[0], sonnet_dists, None),
        "gpt-oss holdout": (load_positions(DB, DEFAULT_MODEL, 0, 45000,
                                           seed_filter=lambda s: s >= HOLDOUT_START)[0], own_dists, 1),
    }

    for set_name, (pos, dists, lm_steps) in sets.items():
        n_all = len(pos)
        pos = [p for p in pos if p["key"] in dists]
        if not pos:
            print(f"{set_name}: no positions with a local-model distribution yet\n")
            continue
        X, y, groups, _ = build_groups(pos)
        s1 = first_step_mask(pos)
        print(f"== {set_name}: {len(pos)} of {n_all} positions (those with a {lm_name} distribution), "
              f"{len(groups)} choice events, {int(s1.sum())} at step 1")
        print(f"   {'model':28s} {'R2 pooled':>10s} {'R2 step 1':>10s} {'R2 steps 2+':>12s}")
        for name, b in boosters.items():
            ll, null = group_log_loss(b.predict(X, raw_score=True), groups, y)
            print(f"   {name:28s} {r2(ll, null):10.4f} {r2(ll, null, s1):10.4f} {r2(ll, null, ~s1):12.4f}")
        ll, null = lm_logloss(pos, dists, lm_steps)
        label = f"{lm_name} raw, T=1"
        if lm_steps == 1:
            print(f"   {label:28s} {'':>10s} {r2(ll, null):10.4f} {'':>12s}   (step 1 only: own-path)")
        else:
            s1_lm = first_step_mask(pos)
            print(f"   {label:28s} {r2(ll, null):10.4f} {r2(ll, null, s1_lm):10.4f} {r2(ll, null, ~s1_lm):12.4f}")
        print()


if __name__ == "__main__":
    main()
