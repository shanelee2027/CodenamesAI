"""Train the online_vocab_listener booster: the incumbent's listener retrained
for the online game's 800 words (docs/versions/online_vocab_listener.md).

**What changes against the incumbent's booster (listener_gbt_assoc_profile.txt).**
- Data: its training rows (train_listener_net.load_sets' `train`) plus the
  online split's generated training positions on 550 of the 800 words
  (`online train`, seeds from 8,000,000).
- Normalisation: every z-scored feature over the online 800
  (load_sets(normalization="online")), the population it is dealt from.
- Features: `--features incumbent` reads the incumbent's 67 columns;
  `--features all` adds the word-kind block (codenames/word_kind.py) and the
  text-embedding block (te_*), 73; `kind` and `text` add one block each. All
  are fitted on identical rows, so the differences between them are the
  blocks alone.

Recipe as the incumbent's: listener_training.train, step weights, early
stopping on validation. The old rows are exactly the incumbent's `train` and
`val` (the base set cache's positions, taken from the online footing by key;
a fresh load holds 848 more, rankings bought since in arena games, and would
also redraw the split). Validation is that `val` plus 15% of the online
training positions, split by board.

Scored, against the incumbent on the same positions on its own (base)
footing: R² pooled, at pick 1 and at picks 2+, with the gain and a 95%
bootstrap over boards. `online held-out words, generated` is the clean test
for both models (250 words neither was trained on). The older sets are
partly contaminated for the new booster: 74 of the 150 old held-out words are
online training words.

    python scripts/pipeline/train_online_vocab_listener.py --features incumbent   # the shipped booster
    python scripts/pipeline/train_online_vocab_listener.py --features all         # + kind + text (73)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

ROOT = Path(__file__).resolve().parents[2]
INCUMBENT = ROOT / "cache" / "listener_gbt_assoc_profile.txt"
# The shipped booster is the incumbent's 67 columns (docs/versions/
# online_vocab_listener.md: the two new blocks raise R² on gpt-oss's sets and
# lower it on Sonnet 5.5's); the others are kept for the comparison.
OUT = {"incumbent": ROOT / "cache" / "listener_gbt_online_vocab.txt",
       "all": ROOT / "cache" / "listener_gbt_online_vocab_kind_text.txt",
       "kind": ROOT / "cache" / "listener_gbt_online_vocab_kind.txt",
       "text": ROOT / "cache" / "listener_gbt_online_vocab_text.txt"}
TEST = ("online held-out words, generated", "held-out words, generated, Sonnet 5.5", "held-out words, generated",
        "online words, generated", "new boards")


def nll_of(booster, positions: list[dict]):
    """Per event: negative log-likelihood, the uniform baseline's, the board
    seed and whether it is a first pick."""
    import codenames.listener_training as T
    from codenames.listener_features import booster_columns

    X, y, g, seeds = T.build_groups(positions)
    sizes = np.asarray(g)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    z = booster.predict(X[:, booster_columns(booster)], raw_score=True)
    gmax = np.maximum.reduceat(z, starts)
    lse = np.log(np.add.reduceat(np.exp(z - np.repeat(gmax, sizes)), starts)) + gmax
    return lse - np.add.reduceat(z * y, starts), np.log(sizes), np.asarray(seeds), T.first_step_mask(positions)


def compare(old, new, base_pos: list[dict], online_pos: list[dict], rng) -> dict:
    """The incumbent on its footing against the new booster on its own, same
    positions in the same order."""
    from train_listener_net import pkey

    # A fresh load can hold a few positions bought since the base cache was
    # built (arena games); both models are scored on the ones in both.
    keys = {pkey(p) for p in online_pos}
    base_pos = [p for p in base_pos if pkey(p) in keys]
    by_key = {pkey(p): p for p in online_pos}
    online_pos = [by_key[pkey(p)] for p in base_pos]
    n0, null, seeds, first = nll_of(old, base_pos)
    n1, _, _, _ = nll_of(new, online_pos)
    _, bid = np.unique(seeds, return_inverse=True)
    nb = bid.max() + 1
    counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
    d = (counts @ np.bincount(bid, n0 - n1, minlength=nb)) / (counts @ np.bincount(bid, null, minlength=nb))
    r2 = lambda nll, m: float(1 - nll[m].sum() / null[m].sum())          # noqa: E731
    every = np.ones_like(first)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"positions": len(base_pos), "events": len(n0),
            "incumbent": [r2(n0, every), r2(n0, first), r2(n0, ~first)],
            "new": [r2(n1, every), r2(n1, first), r2(n1, ~first)],
            "gain": float((n0 - n1).sum() / null.sum()), "ci": [float(lo), float(hi)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(OUT), default="incumbent")
    ap.add_argument("--eval-only", action="store_true", help="score the saved booster without training")
    args = ap.parse_args()

    import lightgbm as lgb

    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES, TEXT_FEATURES
    from codenames.word_kind import KIND_FEATURES
    from eval_listener_accuracy import with_all_columns
    from train_listener_net import ONLINE_TRAIN, load_sets, pkey

    old = lgb.Booster(model_file=str(INCUMBENT))
    block = {"all": list(KIND_FEATURES) + list(TEXT_FEATURES), "incumbent": [], "kind": list(KIND_FEATURES),
             "text": list(TEXT_FEATURES)}[args.features]
    names = old.feature_name() + block
    cols = [FEATURE_NAMES.index(n) for n in names]

    sets, base = load_sets(normalization="online"), load_sets()
    by_key = {pkey(p): p for k in ("train", "val") for p in sets[k]}
    old_tr, old_va = ([by_key[pkey(p)] for p in base[k]] for k in ("train", "val"))
    on_fit, on_val = T.split_positions(sets[ONLINE_TRAIN], 0.15, 0)
    tr, va = old_tr + on_fit, old_va + on_val
    Xtr, ytr, gtr, _ = T.build_groups(tr)
    Xva, yva, gva, _ = T.build_groups(va)
    print(f"{len(names)} features; train {len(tr)} positions ({len(on_fit)} online), {len(gtr)} events; "
          f"val {len(va)} ({len(on_val)} online)", flush=True)
    out = OUT[args.features]
    if args.eval_only:
        b = lgb.Booster(model_file=str(out))
    else:
        b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, 0, names, T.step_weights(tr))
        b.save_model(str(out))
        out.with_suffix(".meta.json").write_text(json.dumps({"normalization": "online"}) + "\n")
    imp = dict(zip(b.feature_name(), b.feature_importance("gain")))
    tot = sum(imp.values())
    new_cols = [c for c in names if c not in old.feature_name()]
    print(f"{b.num_trees()} trees -> {out.name}"
          + (f"; new columns' gain share {sum(imp[c] for c in new_cols) / tot:.1%}" if new_cols else ""))

    rng = np.random.default_rng(0)
    report = {}
    for s in TEST:
        if s not in sets or s not in base:
            continue
        r = report[s] = compare(old, b, with_all_columns(base[s]), sets[s], rng)
        print(f"{s:40s} {r['positions']:5d} pos  incumbent {r['incumbent'][0]:.4f} (pick 1 {r['incumbent'][1]:.4f}, "
              f"2+ {r['incumbent'][2]:.4f})  new {r['new'][0]:.4f} (pick 1 {r['new'][1]:.4f}, 2+ {r['new'][2]:.4f})  "
              f"gain {r['gain']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]", flush=True)
    out.with_suffix(".json").write_text(json.dumps({"features": names, "trees": b.num_trees(), "report": report},
                                                   indent=1))


if __name__ == "__main__":
    main()
