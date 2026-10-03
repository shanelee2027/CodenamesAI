"""The headline listener-accuracy test: McFadden R² by pick on generated
held-out positions (docs/log.md, "A Sonnet copy of the generated held-out
set").

**Why this set.** `held-out words, generated` is positions made exactly like
the training positions (collect_listener_data.py: the same clue mix, reveals
and k) on boards of the 150 held-out words. That makes it the training
distribution on vocabulary no model has seen. Sets of rankings from real games
are easier and skewed, because every clue in them was a spymaster's best.
`held-out words, generated, Sonnet` is the first 913 of the same positions
ranked by Sonnet: a guesser nothing was trained on, so it shows how much of a
gain is the model predicting gpt-oss specifically. `..., Sonnet 5.5` is all of
the positions ranked by Sonnet 5.5, with no unusable answers stored; prefer
it to the Sonnet 5 copy, 8.5% of which is board-order backfill.

Reported per booster (and turn model, if given) on each set:
- R² pooled, with a 95% bootstrap over boards;
- R² at pick 1, 2, 3 and 4+;
- the paired difference from the first booster listed, pooled and at pick 1,
  with a 95% bootstrap over boards (the same events for every booster, so the
  difference is far tighter than either interval).

    python scripts/tools/eval_listener_accuracy.py
    python scripts/tools/eval_listener_accuracy.py --booster incumbent=listener_gbt.txt \\
        --booster "assoc + within-turn=listener_gbt_assoc_features.txt:sequential_listener.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

CACHE = Path(__file__).resolve().parents[2] / "cache"
DEFAULT_BOOSTERS = [
    "incumbent=listener_gbt.txt",
    "conceptnet=listener_gbt_conceptnet.txt",
    "assoc=listener_gbt_assoc_features.txt",
    "assoc_profile=listener_gbt_assoc_profile.txt",
]
PICKS = (("pick 1", lambda s: s == 0), ("pick 2", lambda s: s == 1),
         ("pick 3", lambda s: s == 2), ("pick 4+", lambda s: s >= 3))


def with_all_columns(positions: list[dict]) -> list[dict]:
    """Positions cached before the profile columns existed carry 58 columns;
    append the 11 the way extract() computes them in play (AssocProfile.columns),
    so every booster reads the columns it was fitted on."""
    import codenames.listener_training as T
    from codenames.assoc_profile import PROFILE_FEATURES, AssocProfile
    from codenames.listener_features import N_FEATURES

    short = N_FEATURES - len(PROFILE_FEATURES)
    if all(p["x"].shape[1] == N_FEATURES for p in positions):
        return positions
    profile = AssocProfile.load(T.ASSOC_PROFILE)
    out = []
    for p in positions:
        if p["x"].shape[1] == short:
            p = {**p, "x": np.hstack([p["x"], profile.columns(p["clue"], p["words"])])}
        assert p["x"].shape[1] == N_FEATURES
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", action="append", default=None,
                    help="label=booster file in cache/[:within-turn params file]; repeat. The first is "
                         "the reference for the paired differences")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="train_listener_net.load_sets names; default the three generated held-out copies. "
                         "`train` gives the in-sample fit, for the gap to held-out")
    args = ap.parse_args()

    import lightgbm as lgb

    from codenames.listener_net import WordVectors
    from codenames.sequential_listener import SequentialParams
    from train_joint_listener import Events
    from train_listener_net import GENERATED, GENERATED_SONNET, GENERATED_SONNET55, gbt_scores, load_sets

    specs = []
    for spec in args.booster or DEFAULT_BOOSTERS:
        label, _, files = spec.partition("=")
        path, _, seq = files.partition(":")
        specs.append((label, path, seq or None))

    sets = load_sets()
    vw = WordVectors()
    rng = np.random.default_rng(0)
    for s in args.sets or (GENERATED, GENERATED_SONNET55, GENERATED_SONNET):
        pos = with_all_columns(sets[s])
        e = Events(pos, vw)
        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        draws = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        print(f"\n{s}: {len(pos)} positions, {len(e.step)} events, {nb} boards; events by pick "
              + ", ".join(f"{name} {int(m(e.step).sum())}" for name, m in PICKS))
        print(f"  {'booster':28s} {'R2 [95% CI]':>24s}" + "".join(f"{n:>9s}" for n, _ in PICKS)
              + f"   {'vs ' + specs[0][0] + ', pooled':>30s}   {'pick 1':>24s}")
        ref = None
        for label, path, seq in specs:
            b = lgb.Booster(model_file=str(CACHE / path))
            th = SequentialParams.from_dict(json.loads((CACHE / seq).read_text())) if seq else None
            sc = gbt_scores(b, pos)
            flat = np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)])
            nll, null = e.nll(flat, *e.transform(th, e.drop(sc) if th is not None else None))
            r2 = lambda m: 1 - nll[m].sum() / null[m].sum()
            boot = 1 - (draws @ np.bincount(bid, nll, nb)) / (draws @ np.bincount(bid, null, nb))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            line = f"  {label:28s} {r2(np.ones(len(nll), bool)):8.4f} [{lo:.4f}, {hi:.4f}]"
            line += "".join(f"{r2(m(e.step)):9.4f}" for _, m in PICKS)
            if ref is None:
                ref = nll
            else:
                for m in (np.ones(len(nll), bool), e.step == 0):
                    d = np.bincount(bid, np.where(m, ref - nll, 0.0), nb)
                    den = np.bincount(bid, np.where(m, null, 0.0), nb)
                    db = (draws @ d) / (draws @ den)
                    lo, hi = np.percentile(db, [2.5, 97.5])
                    line += f"   {d.sum() / den.sum():+8.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(line, flush=True)


if __name__ == "__main__":
    main()
