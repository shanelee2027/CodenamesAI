"""A listener that can end its turn: the pick-index booster with a STOP
option, trained on graded relatedness labels (codenames/relatedness.py;
docs/log.md, "Relatedness labels").

**Data.** Each generated training position has a label set S, the words
gpt-oss says the clue points at: "guess" + "stretch" by default (`--set
guess` for the strict set). The words in S are ordered by gpt-oss's stored
ranking of the same position. One choice event per pick:
- pick j (j = 1..|S|): the target is the j-th word of S, chosen among the
  words not yet picked and STOP;
- pick |S| + 1: the target is STOP.
Picks stop at `--depth` (9), as the pick-index booster's do; a set that long
has no STOP event.

**Model.** One booster scores every row of an event: a row per remaining
word, and one STOP row. Softmax over the rows gives the next pick, STOP
included, so the booster learns an absolute level for words against STOP.
That level is what the ranking data never identified. Inputs:
- word rows: the assoc profile booster's columns without `k` (the number the
  ranking prompt was given; these labels never see one, and in play the
  number only caps the turn), plus `x` (the pick number) and `is_stop` = 0;
- the STOP row: only the clue-level columns (the same for every word of a
  position: the top word's lead, the candidate count, the clue's
  association profile), plus `x` and `is_stop` = 1. Its other columns are
  missing, so STOP's score depends on the pick, the clue and how clear the
  board's best word is.

Train and validation are split by board seed (15% validation), within the
labelled positions. Events are weighted equally: unlike the ranking data,
late picks here are the turn ending, which is what this model is for.

    python scripts/pipeline/train_stop_listener.py
    python scripts/pipeline/train_stop_listener.py --set guess
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

CACHE = Path(__file__).resolve().parents[2] / "cache"
BASE_BOOSTER = "listener_gbt_assoc_profile.txt"
DROP = ["k"]
CLUE_LEVEL = ["n_candidates", "peak_z", "lead_margin", "orth_contains", "b_assoc_distinct", "b_assoc_overlap",
              "b_assoc_first", "b_rarity", "b_senses"]
EXTRA = ["x", "is_stop"]
SEED_BASE = 1_000_000
OUT = {"guess+stretch": CACHE / "listener_gbt_stop.txt", "guess": CACHE / "listener_gbt_stop_guess.txt"}


def base_columns() -> tuple[list[str], list[int], list[int]]:
    """(names, their FEATURE_NAMES indices, positions of the clue-level ones)."""
    import lightgbm as lgb

    from codenames.listener_features import FEATURE_NAMES

    names = [n for n in lgb.Booster(model_file=str(CACHE / BASE_BOOSTER)).feature_name() if n not in DROP]
    return names, [FEATURE_NAMES.index(n) for n in names], [names.index(n) for n in CLUE_LEVEL]


def label_sequence(p: dict, guess: list[str], stretch: list[str], use_stretch: bool) -> list[int]:
    """Row indices of the label set, in gpt-oss's ranking order."""
    s = set(guess) | (set(stretch) if use_stretch else set())
    return [t for t in p["targets"] if p["words"][t] in s]


def stop_rows(x: np.ndarray, keep: list[int], pick: int, clue_pos: list[int]) -> np.ndarray:
    """One event's rows: the remaining words, then STOP last."""
    words = np.hstack([x[keep], np.tile([pick, 0.0], (len(keep), 1))])
    stop = np.full((1, x.shape[1] + 2), np.nan)
    stop[0, clue_pos] = x[keep[0], clue_pos]
    stop[0, -2:] = [pick, 1.0]
    return np.vstack([words, stop])


def stop_groups(positions: list[dict], labels: dict, cols: list[int], clue_pos: list[int],
                use_stretch: bool, depth: int = 9):
    """(X, y, groups, per-event info) over every pick of every labelled position."""
    X, y, groups, info = [], [], [], []
    for p in positions:
        lab = labels.get(p["key"])
        if lab is None:
            continue
        seq = label_sequence(p, *lab, use_stretch)
        x = p["x"][:, cols]
        n, taken = p["n"], []
        for j in range(min(len(seq) + 1, depth)):
            keep = [r for r in range(n) if r not in taken]
            if not keep:
                break
            X.append(stop_rows(x, keep, j + 1.0, clue_pos))
            lab_j = np.zeros(len(keep) + 1)
            lab_j[keep.index(seq[j]) if j < len(seq) else len(keep)] = 1.0
            y.append(lab_j)
            groups.append(len(keep) + 1)
            info.append((p["seed"], j, j >= len(seq), len(seq)))
            if j < len(seq):
                taken.append(seq[j])
    return np.vstack(X), np.concatenate(y), groups, info


def load_labels(positions: list[dict], version: str) -> dict:
    from codenames.relatedness import RelatednessStore

    store = RelatednessStore(CACHE / "llm_store.db")
    model = "deepinfra/openai/gpt-oss-120b+effort=low"
    out = {}
    for p in positions:
        got = store.get(model, p["key"][0], list(p["key"][1]), version)
        if got is not None:
            out[p["key"]] = got
    return out


def event_report(b, X, y, groups, info, label: str) -> dict:
    """R² (McFadden, null uniform over the rows), by pick, and how well
    P(STOP) matches when the turn actually ended."""
    s = b.predict(X, raw_score=True)
    starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
    nll, null, p_stop, stopped, pick = [], [], [], [], []
    for st, g, (_, j, is_stop, _) in zip(starts, groups, info):
        z = s[st:st + g]
        lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        nll.append(-lp[np.argmax(y[st:st + g])])
        null.append(np.log(g))
        p_stop.append(np.exp(lp[-1]))
        stopped.append(is_stop)
        pick.append(j)
    nll, null, p_stop, stopped, pick = map(np.asarray, (nll, null, p_stop, stopped, pick))
    r2 = lambda m: 1 - nll[m].sum() / null[m].sum()          # noqa: E731
    out = {"events": len(nll), "r2": r2(pick >= 0)}
    out.update({f"r2_pick_{j + 1}": r2(pick == j) for j in range(4)})
    out["stop_rate"] = float(stopped.mean())
    out["mean_p_stop"] = float(p_stop.mean())
    out["p_stop_when_stopped"] = float(p_stop[stopped].mean())
    out["p_stop_when_continued"] = float(p_stop[~stopped].mean())
    print(f"  {label:10s} {out['events']:6d} events  R2 {out['r2']:.4f}  by pick "
          + " ".join(f"{out[f'r2_pick_{j + 1}']:.3f}" for j in range(4))
          + f"   STOP: actual {out['stop_rate']:.3f}, predicted {out['mean_p_stop']:.3f}; "
          f"P(STOP) {out['p_stop_when_stopped']:.3f} when it stopped, {out['p_stop_when_continued']:.3f} when not")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(OUT), default="guess+stretch")
    ap.add_argument("--version", default="graded-v2")
    ap.add_argument("--depth", type=int, default=9)
    args = ap.parse_args()

    import codenames.listener_training as T
    from eval_listener_accuracy import with_all_columns
    from train_listener_net import load_sets

    names, cols, clue_pos = base_columns()
    train = [p for p in load_sets()["train"] if p["seed"] >= SEED_BASE]
    labels = load_labels(train, args.version)
    pos = with_all_columns([p for p in train if p["key"] in labels])
    fit, val = T.split_positions(pos, 0.15, 0)
    print(f"{len(pos)} labelled positions ({args.version}, set = {args.set}): fit {len(fit)}, val {len(val)}")
    use_stretch = args.set == "guess+stretch"
    Xf, yf, gf, inf = stop_groups(fit, labels, cols, clue_pos, use_stretch, args.depth)
    Xv, yv, gv, inv = stop_groups(val, labels, cols, clue_pos, use_stretch, args.depth)
    sizes = np.array([i[3] for i in inf if i[1] == 0])
    print(f"label set size: mean {sizes.mean():.2f}; " + ", ".join(f"{v}: {(sizes == v).mean():.0%}" for v in range(6))
          + f", 6+: {(sizes >= 6).mean():.0%}")
    b, _ = T.train(Xf, yf, gf, Xv, yv, gv, 8000, 0, names + EXTRA, np.ones(len(gf)))
    print(f"{b.best_iteration} trees")
    report = {"fit": event_report(b, Xf, yf, gf, inf, "fit"), "val": event_report(b, Xv, yv, gv, inv, "val")}
    b.save_model(str(OUT[args.set]), num_iteration=b.best_iteration)
    OUT[args.set].with_suffix(".json").write_text(json.dumps(
        {"version": args.version, "set": args.set, "depth": args.depth, "clue_level": CLUE_LEVEL, "dropped": DROP,
         "trees": b.best_iteration, "report": report}, indent=1))
    print(f"saved -> {OUT[args.set]}")


if __name__ == "__main__":
    main()
