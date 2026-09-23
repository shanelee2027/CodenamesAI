"""Fit the listener with decoys in the denominator, and test the level for D.

`analyze_decoy_invariance.py` showed the NONPARAMETRIC level -- decoy-first
events over chance -- is 0.50 at D=2, 5 and 10. That is necessary but not
sufficient: what the spymaster will actually consume is a fitted score, and a
score can drift with D even when a count statistic does not. This fits the
model and re-runs the same question on fitted quantities.

Two readings of "the level", both reported:

  nats    MEAN decoy score minus the best board score, per position: how far
          below the board's best a random word sits, in the units the softmax
          works in. Under IIA it is a property of the clue and board, so it
          must not move with D.

          It is deliberately a mean and not a logsumexp. The softmax
          denominator carries the TOTAL decoy mass, which grows like log(D)
          whatever the model does -- and logsumexp is dominated by the largest
          of D draws, so even dividing that out leaves an extreme-value trend.
          Neither is a defect in the fit, and a statistic carrying either
          cannot test invariance. The mean has no such term.

  calib   the model's predicted P(first pick is a decoy) against the observed
          rate, broken out by D. A single fitted function that reproduces the
          observed rate at every D -- without D being a feature -- is the
          operational form of invariance.

**Truncation, not step decay.** The main pipeline down-weights deep steps by
0.75**j because the teacher's ranking tail is arbitrary once it has taken the
words it wants. Truncating at the first decoy already removes that tail, and
the decoy term sits at the DEEPEST kept step, so applying the decay here would
down-weight precisely the observation being collected. Weights are uniform.

**`n_candidates` is overwritten with the board count.** extract() computes it
over whatever list it is given, so leaving it alone would let the decoy count
leak in as a feature and the model could read D directly -- which would make
the invariance test vacuous. The feature describes the board; decoys are extra
alternatives, not board words.

Usage:
    python scripts/tools/fit_decoy_level.py cache/training_data/decoys.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats
from codenames.listener_features import FEATURE_NAMES, extract
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

import codenames.listener_training as _tl

NCAND = FEATURE_NAMES.index("n_candidates")


def load_positions(path: Path, sims, stats, aux) -> tuple[list[dict], dict]:
    clue_index = {w.lower(): i for i, w in enumerate(stats.clue_words)}
    out, dropped = [], {"features": 0, "short": 0, "clue_oov": 0}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ranking"):
            dropped["short"] += 1
            continue
        cand = r["candidates"]
        # Shuffle before extraction, as the main pipeline does: the teacher's
        # own order would put every target at a fixed index and turn any
        # positional artifact into the answer.
        perm = random.Random(f"{r['seed']}|{r['clue']}").sample(range(len(cand)), len(cand))
        shuffled = [cand[i] for i in perm]
        where = {w: i for i, w in enumerate(shuffled)}

        n_steps = min(r["cut"] + 1, len(shuffled) - 1)
        picks = [w for w in r["ranking"][:n_steps] if w in where]
        if len(picks) < n_steps or n_steps < 1:
            dropped["short"] += 1
            continue

        feats = extract(r["clue"], shuffled, r["number"], sims, stats, aux["wstats"],
                        clue_index, aux["swow"], aux["entity"], aux["pmi"], aux["extra"],
                        aux["norms"], aux["wordnet"], aux["lexical"])
        if feats is None:
            dropped["features"] += 1
            continue
        feats = np.array(feats, dtype=np.float64)
        feats[:, NCAND] = r["n_board"]          # decoys are not board words

        out.append({"seed": r["seed"], "x": feats, "n": len(shuffled),
                    "targets": [where[w] for w in picks],
                    "decoy_rows": np.array([where[w] for w in r["decoys"] if w in where]),
                    "n_decoys": r["n_decoys"], "n_board": r["n_board"]})
    return out, dropped


def build_groups(positions: list[dict]):
    """PL expansion truncated at the first decoy. Step j is a group over the
    words not yet picked; the final step's target IS the decoy."""
    X, y, groups, seeds = [], [], [], []
    for p in positions:
        taken: list[int] = []
        for j, tgt in enumerate(p["targets"]):
            keep = [r for r in range(p["n"]) if r not in taken]
            rows = p["x"][keep]
            lab = np.zeros(len(rows))
            lab[keep.index(tgt)] = 1.0
            X.append(rows); y.append(lab); groups.append(len(rows)); seeds.append(p["seed"])
            taken.append(tgt)
    return np.vstack(X), np.concatenate(y), groups, np.array(seeds)


def level_nats(booster, p: dict) -> float | None:
    """mean(decoy scores) - max(board score), at step 0. See the docstring on
    why this is a mean rather than a logsumexp."""
    if len(p["decoy_rows"]) == 0:
        return None
    s = booster.predict(p["x"], raw_score=True)
    board = np.setdiff1d(np.arange(p["n"]), p["decoy_rows"])
    return float(s[p["decoy_rows"]].mean() - s[board].max())


def p_decoy_first(booster, p: dict) -> float:
    s = booster.predict(p["x"], raw_score=True)
    e = np.exp(s - s.max())
    return float(e[p["decoy_rows"]].sum() / e.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.3)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    aux = {
        "wstats": _tl.WordStats.load(_tl.WORD_STATS),
        "swow": _tl.SwowTables.load(_tl.SWOW_TABLES) if _tl.SWOW_TABLES.exists() else None,
        "entity": _tl.EntitySims.load(_tl.ENTITY_SIMS) if _tl.ENTITY_SIMS.exists() else None,
        "pmi": _tl.EntitySims.load(_tl.LM_PMI) if _tl.LM_PMI.exists() else None,
        "extra": _tl.ExtraSims.load(_tl.EXTRA_SIMS) if _tl.EXTRA_SIMS.exists() else None,
        "norms": _tl.WordNorms.load(_tl.WORD_NORMS) if _tl.WORD_NORMS.exists() else None,
        "wordnet": _tl.ExtraSims.load(_tl.WORDNET_SIMS) if _tl.WORDNET_SIMS.exists() else None,
        "lexical": _tl.ExtraSims.load(_tl.LEXICAL_SIMS) if _tl.LEXICAL_SIMS.exists() else None,
    }

    pos, dropped = load_positions(args.path, sims, stats, aux)
    print(f"{len(pos)} positions   dropped: {dropped}")

    # Split by board seed, never by row: one position yields several groups.
    seeds = sorted({p["seed"] for p in pos})
    random.Random(0).shuffle(seeds)
    val = set(seeds[: int(args.val_frac * len(seeds))])
    tr = [p for p in pos if p["seed"] not in val]
    va = [p for p in pos if p["seed"] in val]
    print(f"train {len(tr)} positions / val {len(va)}")

    Xtr, ytr, gtr, _ = build_groups(tr)
    Xva, yva, gva, _ = build_groups(va)
    print(f"groups: {len(gtr)} train / {len(gva)} val   rows: {len(Xtr)} / {len(Xva)}")

    booster, acc = _tl.train(Xtr, ytr, gtr, Xva, yva, gva, args.rounds)
    preds = booster.predict(Xva, raw_score=True)
    r2 = _tl.mcfadden_on(preds, gva, yva)
    print(f"\nval McFadden R2 {r2:.4f}   top-1 accuracy {acc:.3f}"
          f"   best iter {booster.best_iteration}")

    print("\n  D     n    level (nats)      predicted   observed    gap")
    print("  " + "-" * 56)
    for d in sorted({p["n_decoys"] for p in va}):
        grp = [p for p in va if p["n_decoys"] == d]
        lv = [x for x in (level_nats(booster, p) for p in grp) if x is not None]
        pr = [p_decoy_first(booster, p) for p in grp]
        ob = [float(p["targets"][0] in set(p["decoy_rows"].tolist())) for p in grp]
        print(f" {d:2d} {len(grp):5d} {np.mean(lv):9.2f} +-{np.std(lv)/np.sqrt(len(lv)):.2f}"
              f" {np.mean(pr):12.3f} {np.mean(ob):10.3f} {np.mean(pr)-np.mean(ob):+7.3f}")
    print("\nD is not a feature, so a level flat across D says the fitted scale is a")
    print("property of the clue and board; a trend says decoys interact after all.")
    print("The calibration columns are the operational test: one fitted function")
    print("reproducing the observed decoy-first rate at every D.")


if __name__ == "__main__":
    main()
