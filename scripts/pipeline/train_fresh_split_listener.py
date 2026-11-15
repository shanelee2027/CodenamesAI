"""Train the fresh_split_listener booster: the GBT listener refitted from
scratch on the three-way split of the online game's 800 words
(docs/versions/fresh_split_listener.md; board.py::load_three_way_split).

**Data.** Only the positions generated on the split (scripts/data/
collect_listener_data.py --vocab split_train / split_tune / split_test, seeds
from listener_training.SPLIT_BASES); nothing collected before it. A position
is kept only if it is exactly what the generator planned for its board
(clue and candidates, and the number for numbered rankings), so an older
position whose few remaining words happen to sit on a new board can never be
picked up by the word-set lookup.

**Two prompts** (`--prompt`), the same planned positions under each:
- `numbered`: gpt-oss was shown the clue with the planned number (1-4) and had
  to name at least number + 2 words; the picks used are the first `number`.
- `plain` (--no-number): the clue alone, with a full ranking (at least 80% of
  the words named); the picks used are the first n, n = own words still
  unrevealed on the planned board (at most the guaranteed depth). Positions
  with no own word left are dropped.

**Roles of the three parts.**
- train (400 words): the boosters are fitted here; 15% of its boards, split by
  board, are the early-stopping set.
- tune (150 words): variants and settings are compared here, by McFadden R²
  pooled and by pick (1, 2-4, 5+), each against a reference with a 95%
  bootstrap over boards. Everything is chosen on this set only.
- test (250 words): loaded only with `--test`, once the choice is final.

**Variants** (identical rows, identical settings; the difference is the
columns): `full` is every feature except the blocks earlier models measured
and left out (sense_*, the two-hop med_*), 73 columns. `no_assoc` drops every
column built from gpt-oss's free-association lists (assoc_share, assoc_rank,
the b_assoc_* clue profile, the c_rev_* reverse lists, hub_assoc), the only
features that need prompting. `no_pmi` drops pmi_gaptop (GPT-2 PMI),
`no_text` the te_* text-embedding block, `no_extras` all three. `no_k` drops
the clue's number (for `plain` rankings it is a constant anyway). On numbered
data `full` is also scored with `k` set to 4 on every position (`full@k=4`),
what the spymaster fed it in play.

`--decay` is the weight of pick j relative to pick j-1 in the fit (0.75 is
listener_training.STEP_DECAY, what every earlier booster used); scoring is
unweighted.

Features are z-scored over the online 800 (normalization "online"). That
standardisation uses no guesser answers, only the similarity tables.

    python scripts/pipeline/train_fresh_split_listener.py --prepare                        # numbered train + tune
    python scripts/pipeline/train_fresh_split_listener.py --variant full                   # fit one variant
    python scripts/pipeline/train_fresh_split_listener.py --score                          # tune-set table
    python scripts/pipeline/train_fresh_split_listener.py --prompt plain --prepare
    python scripts/pipeline/train_fresh_split_listener.py --prompt plain --variant no_k --decay 0.6
    python scripts/pipeline/train_fresh_split_listener.py --prompt plain --score --eval-on numbered
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_DECAY = 0.75
# A full ranking only has to name 80% of the words (guessers/llm.py::required_named).
FULL_COVERAGE = 0.8


def variants() -> dict[str, list[str]]:
    from codenames.listener_features import FEATURE_NAMES, MEDIATED_FEATURES, TEXT_FEATURES

    full = [f for f in FEATURE_NAMES if f not in ("sense_agree", "sense_spread", *MEDIATED_FEATURES)]
    drop = {"full": (), "no_assoc": ASSOC, "no_pmi": PMI, "no_text": TEXT_FEATURES,
            "no_extras": ASSOC + PMI + tuple(TEXT_FEATURES), "no_k": ("k",)}
    return {name: [f for f in full if f not in d] for name, d in drop.items()}


def tag(variant: str, prompt: str, decay: float) -> str:
    """The numbered prompt at the default decay keeps the first run's names."""
    if prompt == "numbered" and decay == DEFAULT_DECAY:
        return variant
    return f"{prompt}_{variant}_d{decay:g}"


def booster_path(variant: str, prompt: str = "numbered", decay: float = DEFAULT_DECAY) -> Path:
    return CACHE / f"listener_gbt_fresh_split_{tag(variant, prompt, decay)}.txt"


def planned(part: str) -> dict[tuple, tuple[int, int]]:
    """(clue, sorted candidates) -> (planned number, own words unrevealed) for
    every position the collector planned for this part, regenerated from its
    deterministic plan."""
    from collect_listener_data import DEFAULT_MIX, build_clue_pool, make_position

    from codenames.board import Board, Role, load_three_way_split
    from codenames.clue_stats import ClueStats
    from codenames.listener_training import SPLIT_BASES
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    pool = build_clue_pool(stats, 10.0)
    space_i = sims.spaces.index("numberbatch")
    vocab = load_three_way_split()[part]
    out = {}
    for i in range(N_PLANNED[part]):
        p = make_position(i, sims, stats, pool, space_i, DEFAULT_MIX, vocab, SPLIT_BASES[part])
        if p is None:
            continue
        own = set(Board.generate(seed=SPLIT_BASES[part] + i, vocabulary=vocab).words_by_role(Role.OWN))
        out[(p["clue"], tuple(sorted(w.lower() for w in p["candidates"])))] = (
            p["k"], sum(w in own for w in p["candidates"]))
    return out


def load_part(part: str, prompt: str) -> list[dict]:
    import codenames.listener_training as T

    base = T.SPLIT_BASES[part]
    pos, _ = T.load_positions(T.DB, T.DEFAULT_MODEL, 0, 0, split={part: N_PLANNED[part]},
                              seed_filter=lambda s: base <= s < base + 1_000_000, normalization="online",
                              no_number=prompt == "plain")
    plan = planned(part)
    kept, unplanned, no_own = [], 0, 0
    for p in pos:
        hit = plan.get((p["clue"], tuple(sorted(w.lower() for w in p["key"][1]))))
        if hit is None or (prompt == "numbered" and hit[0] != p["k"]):
            unplanned += 1
            continue
        if prompt == "plain":
            n_own = hit[1]
            if n_own == 0:
                no_own += 1
                continue
            p["k"] = min(n_own, math.ceil(FULL_COVERAGE * p["n"]))
        kept.append(p)
    print(f"{prompt} {part}: {len(kept)} positions on {len({p['seed'] for p in kept})} boards "
          f"({unplanned} resolved to a split board but not planned there; {no_own} with no own word left; "
          f"{len(plan)} planned)", flush=True)
    return kept


def load_sets(parts=("train", "tune"), prompt: str = "numbered") -> dict[str, list[dict]]:
    sets = pickle.loads(SETS.read_bytes()) if SETS.exists() else {}
    name = (lambda part: part) if prompt == "numbered" else (lambda part: f"{prompt} {part}")
    missing = [p for p in parts if name(p) not in sets]
    for part in missing:
        sets[name(part)] = load_part(part, prompt)
    if missing:
        SETS.write_bytes(pickle.dumps(sets, protocol=pickle.HIGHEST_PROTOCOL))
    return {p: sets[name(p)] for p in parts}


def fit(variant: str, seed: int, prompt: str, decay: float) -> None:
    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES

    names = variants()[variant]
    cols = [FEATURE_NAMES.index(n) for n in names]
    train = load_sets(("train",), prompt)["train"]
    tr, va = T.split_positions(train, 0.15, 0)
    Xtr, ytr, gtr, _ = T.build_groups(tr)
    Xva, yva, gva, _ = T.build_groups(va)
    label = tag(variant, prompt, decay)
    print(f"{label}: {len(names)} features; fit {len(tr)} positions, {len(gtr)} events; "
          f"early stopping on {len(va)} positions", flush=True)
    b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, seed, names, T.step_weights(tr, decay))
    out = booster_path(variant, prompt, decay)
    b.save_model(str(out))
    out.with_suffix(".meta.json").write_text(json.dumps({"normalization": "online"}) + "\n")
    print(f"{label}: {b.num_trees()} trees -> {out.name}", flush=True)


def nll_of(booster, positions: list[dict], k_as: int | None = None):
    """Per event: negative log-likelihood, the uniform baseline's, the board
    seed and the pick index (1-based). `k_as` overrides the `k` feature."""
    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES, booster_columns

    X, y, g, seeds = T.build_groups(positions)
    if k_as is not None:
        X = X.copy()
        X[:, FEATURE_NAMES.index("k")] = k_as
    sizes = np.asarray(g)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    z = booster.predict(X[:, booster_columns(booster)], raw_score=True)
    gmax = np.maximum.reduceat(z, starts)
    lse = np.log(np.add.reduceat(np.exp(z - np.repeat(gmax, sizes)), starts)) + gmax
    pick = np.array([j + 1 for p in positions for j in range(min(p["k"], p["n"] - 1))])
    return lse - np.add.reduceat(z * y, starts), np.log(sizes), np.asarray(seeds), pick


def fitted_boosters(prompt: str) -> dict[str, Path]:
    """label -> booster file, for every variant and decay fitted on `prompt`."""
    if prompt == "numbered":
        return {v: booster_path(v) for v in variants() if booster_path(v).exists()}
    return {p.stem.removeprefix(f"listener_gbt_fresh_split_{prompt}_"): p
            for p in sorted(CACHE.glob(f"listener_gbt_fresh_split_{prompt}_*.txt"))}


def score(part: str, prompt: str, eval_on: str | None = None, reference: str | None = None) -> dict:
    """R² of every booster fitted on `prompt`, on `part` of the `eval_on`
    rankings (default: the same prompt), and each one's difference from
    `reference` with a 95% bootstrap over boards (2,000 resamples)."""
    import lightgbm as lgb

    eval_on = eval_on or prompt
    positions = load_sets((part,), eval_on)[part]
    files = fitted_boosters(prompt)
    reference = reference or ("full" if prompt == "numbered" else f"no_k_d{DEFAULT_DECAY:g}")
    nll = {}
    for v, f in files.items():
        nll[v], null, seeds, pick = nll_of(lgb.Booster(model_file=str(f)), positions)
    if prompt == "numbered" and eval_on == "numbered" and "full" in files:
        nll["full@k=4"] = nll_of(lgb.Booster(model_file=str(files["full"])), positions, k_as=4)[0]
    _, bid = np.unique(seeds, return_inverse=True)
    nb = bid.max() + 1
    rng = np.random.default_rng(0)
    counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
    groups = {"all": np.ones(len(null), bool), "1": pick == 1, "2-4": (pick >= 2) & (pick <= 4), "5+": pick >= 5}
    groups = {k: m for k, m in groups.items() if m.any()}
    r2 = lambda x, m: float(1 - x[m].sum() / null[m].sum())             # noqa: E731
    print(f"boosters fitted on {prompt} rankings, scored on {eval_on} {part}: {len(positions)} positions, "
          f"{len(null)} choice events ({', '.join(f'{k} {m.sum()}' for k, m in groups.items())}), {nb} boards")
    print(f"  {'booster':16s} " + " ".join(f"{k:>7s}" for k in groups) + f"   minus {reference} (95% CI over boards)")
    report = {}
    for v in nll:
        row = report[v] = {k: r2(nll[v], m) for k, m in groups.items()}
        line = f"  {v:16s} " + " ".join(f"{row[k]:7.4f}" for k in groups)
        if v != reference and reference in nll:
            out = []
            for name, m in groups.items():
                diff = np.where(m, nll[reference] - nll[v], 0.0)
                boot = (counts @ np.bincount(bid, diff, minlength=nb)) / (
                    counts @ np.bincount(bid, np.where(m, null, 0.0), minlength=nb))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                est = float(diff.sum() / null[m].sum())
                row[f"minus {reference} {name}"] = [est, float(lo), float(hi)]
                out.append(f"{name} {est:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            line += "   " + "; ".join(out)
        print(line, flush=True)
    (CACHE / f"fresh_split_scores_{prompt}_on_{eval_on}_{part}.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", choices=("numbered", "plain"), default="numbered",
                    help="which rankings to fit on: the clue with its number, or the clue alone")
    ap.add_argument("--prepare", action="store_true", help="load train and tune positions and cache them")
    ap.add_argument("--variant", choices=list(variants()), help="fit this variant")
    ap.add_argument("--decay", type=float, default=DEFAULT_DECAY, help="weight of pick j over pick j-1 in the fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="score every fitted booster on the tuning set")
    ap.add_argument("--eval-on", choices=("numbered", "plain"), default=None,
                    help="score on these rankings (default: --prompt's)")
    ap.add_argument("--reference", default=None, help="booster label the differences are taken against")
    ap.add_argument("--test", action="store_true",
                    help="score on the TEST set; only once the variant and settings are final")
    args = ap.parse_args()
    if args.prepare:
        load_sets(("train", "tune"), args.prompt)
    if args.variant:
        fit(args.variant, args.seed, args.prompt, args.decay)
    if args.score:
        score("tune", args.prompt, args.eval_on, args.reference)
    if args.test:
        score("test", args.prompt, args.eval_on, args.reference)


if __name__ == "__main__":
    main()
