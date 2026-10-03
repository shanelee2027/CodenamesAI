"""Train the assoc_profile_listener booster: the assoc booster's 56 features
plus the clue profile (B) and reverse associations (C) from
codenames/assoc_profile.py (docs/versions/assoc_profile_listener.md).

Same rows, split and recipe as the assoc booster (listener_training.train on
train_listener_net.load_sets' train split, early-stopped on val, step
weights). The cached positions predate the new columns, so they are appended
here with AssocProfile.columns, the same function extract() calls in play.

Scored like every listener table: R² pooled, at pick 1 and at picks 2+
(frozen Plackett-Luce), with the gain over the assoc booster and a 95%
bootstrap over boards.

    python scripts/pipeline/train_profile_listener.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "cache" / "listener_gbt_assoc_features.txt"
OUT = ROOT / "cache" / "listener_gbt_assoc_profile.txt"
SETS = ("val", "new boards", "held-out words, generated", "held-out words", "held-out words, Sonnet")


def with_profile(positions: list[dict], profile) -> list[dict]:
    from codenames.assoc_profile import PROFILE_FEATURES
    from codenames.listener_features import N_FEATURES

    out = []
    for p in positions:
        assert p["x"].shape[1] == N_FEATURES - len(PROFILE_FEATURES), "positions already carry the new columns"
        out.append({**p, "x": np.hstack([p["x"], profile.columns(p["clue"], p["words"])])})
    return out


def scores(booster, X, starts, sizes, y):
    """Per event: negative log-likelihood and the uniform baseline's."""
    z = booster.predict(X, raw_score=True)
    gmax = np.maximum.reduceat(z, starts)
    lse = np.log(np.add.reduceat(np.exp(z - np.repeat(gmax, sizes)), starts)) + gmax
    return lse - np.add.reduceat(z * y, starts), np.log(sizes)


def main() -> None:
    import lightgbm as lgb

    import codenames.listener_training as T
    from codenames.assoc_profile import PROFILE_FEATURES, AssocProfile
    from codenames.listener_features import FEATURE_NAMES
    from train_listener_net import load_sets

    profile = AssocProfile.load(T.ASSOC_PROFILE)
    sets = load_sets()
    base = lgb.Booster(model_file=str(BASE))
    names = base.feature_name() + list(PROFILE_FEATURES)
    cols = [FEATURE_NAMES.index(n) for n in names]
    base_cols = [FEATURE_NAMES.index(n) for n in base.feature_name()]

    tr, va = with_profile(sets["train"], profile), with_profile(sets["val"], profile)
    Xtr, ytr, gtr, _ = T.build_groups(tr)
    Xva, yva, gva, _ = T.build_groups(va)
    print(f"{len(names)} features, {len(gtr)} train events", flush=True)
    b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, 0, names, T.step_weights(sets["train"]))
    b.save_model(str(OUT))
    imp = dict(zip(b.feature_name(), b.feature_importance("gain")))
    tot = sum(imp.values())
    print(f"{b.num_trees()} trees -> {OUT.name}; new features' gain share "
          f"{sum(imp[c] for c in PROFILE_FEATURES) / tot:.1%}")

    rng = np.random.default_rng(0)
    for s in SETS:
        ps = with_profile(sets[s], profile)
        X, y, g, seeds = T.build_groups(ps)
        sizes = np.asarray(g)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        first = T.first_step_mask(sets[s])
        _, bid = np.unique(seeds, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        nll0, null = scores(base, X[:, base_cols], starts, sizes, y)
        nll1, _ = scores(b, X[:, cols], starts, sizes, y)
        r2 = lambda nll, m: 1 - nll[m].mean() / null[m].mean()
        allm = np.ones_like(first)
        d = (counts @ np.bincount(bid, nll0 - nll1, minlength=nb)) / (counts @ np.bincount(bid, null, minlength=nb))
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{s:27s} assoc {r2(nll0, allm):.4f} (pick 1 {r2(nll0, first):.4f})  "
              f"+profile {r2(nll1, allm):.4f} (pick 1 {r2(nll1, first):.4f}, 2+ {r2(nll1, ~first):.4f})  "
              f"gain {(nll0 - nll1).sum() / null.sum():+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
