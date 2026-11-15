"""Train the fresh_split_listener booster: the GBT listener refitted from
scratch on the three-way split of the online game's 800 words
(docs/versions/fresh_split_listener.md; board.py::load_three_way_split).

**Data.** Only the positions generated on the split (scripts/data/
collect_listener_data.py --vocab split_train / split_tune / split_test, seeds
from listener_training.SPLIT_BASES); nothing collected before it. A position
is kept only if it is exactly what the generator planned for its board
(clue, candidates and number), so an older position whose few remaining words
happen to sit on a new board can never be picked up by the word-set lookup.

**Roles of the three parts.**
- train (400 words): the boosters are fitted here; 15% of its boards, split by
  board, are the early-stopping set.
- tune (150 words): the variants below are compared here, by McFadden R²
  pooled, at pick 1 and at picks 2+, each against `full` with a 95% bootstrap
  over boards. Feature blocks and settings are chosen on this set only.
- test (250 words): loaded only with `--test`, once the choice is final.

**Variants** (identical rows, identical settings; the difference is the
columns): `full` is every feature except the blocks earlier models measured
and left out (sense_*, the two-hop med_*), 73 columns. `no_assoc` drops every
column built from gpt-oss's free-association lists (assoc_share, assoc_rank,
the b_assoc_* clue profile, the c_rev_* reverse lists, hub_assoc), the only
features that need prompting. `no_pmi` drops pmi_gaptop (GPT-2 PMI),
`no_text` the te_* text-embedding block, `no_extras` all three.

Features are z-scored over the online 800 (normalization "online"). That
standardisation uses no guesser answers, only the similarity tables.

    python scripts/pipeline/train_fresh_split_listener.py --prepare          # load and cache train + tune
    python scripts/pipeline/train_fresh_split_listener.py --variant full     # fit one variant
    python scripts/pipeline/train_fresh_split_listener.py --score            # tune-set table
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
SETS = CACHE / "fresh_split_sets.pkl"
# Positions planned per part, as collected (the generator drops ~12% of plans).
N_PLANNED = {"train": 34000, "tune": 3400, "test": 2300}
ASSOC = ("assoc_share", "assoc_rank", "b_assoc_distinct", "b_assoc_overlap", "b_assoc_first",
         "c_rev_share", "c_rev_rank", "c_rev_board_rank", "hub_assoc")
PMI = ("pmi_gaptop",)


def variants() -> dict[str, list[str]]:
    from codenames.listener_features import FEATURE_NAMES, MEDIATED_FEATURES, TEXT_FEATURES

    full = [f for f in FEATURE_NAMES if f not in ("sense_agree", "sense_spread", *MEDIATED_FEATURES)]
    drop = {"full": (), "no_assoc": ASSOC, "no_pmi": PMI, "no_text": TEXT_FEATURES,
            "no_extras": ASSOC + PMI + tuple(TEXT_FEATURES)}
    return {name: [f for f in full if f not in d] for name, d in drop.items()}


def booster_path(variant: str) -> Path:
    return CACHE / f"listener_gbt_fresh_split_{variant}.txt"


def planned_keys(part: str) -> set[tuple]:
    """(clue, sorted candidates, number) of every position the collector
    planned for this part, regenerated from its deterministic plan."""
    from collect_listener_data import DEFAULT_MIX, build_clue_pool, make_position

    from codenames.board import load_three_way_split
    from codenames.clue_stats import ClueStats
    from codenames.listener_training import SPLIT_BASES
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    pool = build_clue_pool(stats, 10.0)
    space_i = sims.spaces.index("numberbatch")
    vocab = load_three_way_split()[part]
    keys = set()
    for i in range(N_PLANNED[part]):
        p = make_position(i, sims, stats, pool, space_i, DEFAULT_MIX, vocab, SPLIT_BASES[part])
        if p is not None:
            keys.add((p["clue"], tuple(sorted(w.lower() for w in p["candidates"])), p["k"]))
    return keys


def load_part(part: str) -> list[dict]:
    import codenames.listener_training as T

    base = T.SPLIT_BASES[part]
    pos, dropped = T.load_positions(T.DB, T.DEFAULT_MODEL, 0, 0, split={part: N_PLANNED[part]},
                                    seed_filter=lambda s: base <= s < base + 1_000_000, normalization="online")
    keys = planned_keys(part)
    kept = [p for p in pos if (p["clue"], tuple(sorted(w.lower() for w in p["key"][1])), p["k"]) in keys]
    print(f"{part}: {len(kept)} positions on {len({p['seed'] for p in kept})} boards "
          f"({len(pos) - len(kept)} resolved to a split board but not planned there; {len(keys)} planned)",
          flush=True)
    return kept


def load_sets(parts=("train", "tune")) -> dict[str, list[dict]]:
    sets = pickle.loads(SETS.read_bytes()) if SETS.exists() else {}
    missing = [p for p in parts if p not in sets]
    for part in missing:
        sets[part] = load_part(part)
    if missing:
        SETS.write_bytes(pickle.dumps(sets, protocol=pickle.HIGHEST_PROTOCOL))
    return {p: sets[p] for p in parts}


def fit(variant: str, seed: int) -> None:
    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES

    names = variants()[variant]
    cols = [FEATURE_NAMES.index(n) for n in names]
    train = load_sets(("train",))["train"]
    tr, va = T.split_positions(train, 0.15, 0)
    Xtr, ytr, gtr, _ = T.build_groups(tr)
    Xva, yva, gva, _ = T.build_groups(va)
    print(f"{variant}: {len(names)} features; fit {len(tr)} positions, {len(gtr)} events; "
          f"early stopping on {len(va)} positions", flush=True)
    b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, seed, names, T.step_weights(tr))
    out = booster_path(variant)
    b.save_model(str(out))
    out.with_suffix(".meta.json").write_text(json.dumps({"normalization": "online"}) + "\n")
    print(f"{variant}: {b.num_trees()} trees -> {out.name}", flush=True)


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


def score(part: str, reference: str = "full") -> dict:
    """R² of every fitted variant on `part`, and each one's difference from
    `reference` with a 95% bootstrap over boards (2,000 resamples)."""
    import lightgbm as lgb

    positions = load_sets((part,))[part]
    fitted = [v for v in variants() if booster_path(v).exists()]
    nll = {}
    for v in fitted:
        nll[v], null, seeds, first = nll_of(lgb.Booster(model_file=str(booster_path(v))), positions)
    _, bid = np.unique(seeds, return_inverse=True)
    nb = bid.max() + 1
    rng = np.random.default_rng(0)
    counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
    r2 = lambda x, m: float(1 - x[m].sum() / null[m].sum())             # noqa: E731
    every = np.ones_like(first)
    print(f"{part}: {len(positions)} positions, {len(null)} choice events, {nb} boards")
    print(f"  {'variant':10s} {'R2':>7s} {'pick 1':>7s} {'2+':>7s}   minus {reference} (95% CI over boards)")
    report = {}
    for v in fitted:
        row = report[v] = {"r2": [r2(nll[v], every), r2(nll[v], first), r2(nll[v], ~first)]}
        line = f"  {v:10s} {row['r2'][0]:7.4f} {row['r2'][1]:7.4f} {row['r2'][2]:7.4f}"
        if v != reference and reference in nll:
            out = []
            for name, m in (("all", every), ("pick 1", first), ("2+", ~first)):
                diff = np.where(m, nll[reference] - nll[v], 0.0)
                boot = (counts @ np.bincount(bid, diff, minlength=nb)) / (counts @ np.bincount(bid, np.where(m, null, 0.0), minlength=nb))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                est = float(diff.sum() / null[m].sum())
                row[name] = [est, float(lo), float(hi)]
                out.append(f"{name} {est:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            line += "   " + "; ".join(out)
        print(line, flush=True)
    (CACHE / f"fresh_split_scores_{part}.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="load train and tune positions and cache them")
    ap.add_argument("--variant", choices=list(variants()), help="fit this variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="score every fitted variant on the tuning set")
    ap.add_argument("--test", action="store_true",
                    help="score on the TEST set; only once the variant and settings are final")
    args = ap.parse_args()
    if args.prepare:
        load_sets(("train", "tune"))
    if args.variant:
        fit(args.variant, args.seed)
    if args.score:
        score("tune")
    if args.test:
        score("test")


if __name__ == "__main__":
    main()
