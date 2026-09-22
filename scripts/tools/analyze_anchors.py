"""Summarise the anchor probe: does either anchor move with clue quality?

Reads probe_anchors.py's output and answers one question per anchor -- does it
separate the three clue grades? A flat anchor is inert and the re-collection it
would cost is wasted, so the test is deliberately blunt: the grades must order
correctly AND the gap must survive a permutation test, because with 40 boards
an eyeballed difference of a few percent means nothing.

Paired by board, because clue grade varies within a board and board difficulty
varies between them; an unpaired comparison would mostly measure the boards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRADES = ("top", "mid", "random")


def paired(rows, metric, a, b):
    """(a - b) per board, where both grades produced a usable number."""
    by_seed = {}
    for r in rows:
        if r.get(metric) is not None:
            by_seed.setdefault(r["seed"], {})[r["grade"]] = r[metric]
    return [v[a] - v[b] for v in by_seed.values() if a in v and b in v]


def perm_p(diffs, trials=20000, seed=7):
    """Two-sided sign-flip test on the paired differences: under the null the
    sign of each board's difference is a coin flip. Vectorised over trials --
    a Python loop here made the whole summary take longer than the probe."""
    if not diffs:
        return float("nan")
    d = np.asarray(diffs, dtype=float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(trials, d.size))
    hits = int((np.abs((signs * d).mean(axis=1)) >= obs).sum())
    return (hits + 1) / (trials + 1)


def describe(rows, metric, label, higher_means_more_confident):
    print(f"\n=== {label} ===")
    for g in GRADES:
        vals = [r[metric] for r in rows if r["grade"] == g and r.get(metric) is not None]
        n_missing = sum(1 for r in rows if r["grade"] == g and r.get(metric) is None)
        if vals:
            print(f"  {g:<7} n={len(vals):<3} mean {np.mean(vals):.3f}  median {np.median(vals):.3f}"
                  f"  sd {np.std(vals):.3f}" + (f"   ({n_missing} unusable)" if n_missing else ""))
    for a, b in (("top", "random"), ("top", "mid"), ("mid", "random")):
        d = paired(rows, metric, a, b)
        if d:
            direction = "more" if (np.mean(d) > 0) == higher_means_more_confident else "LESS"
            print(f"  {a} - {b}: mean {np.mean(d):+.3f} over {len(d)} boards, "
                  f"p = {perm_p(d):.4f}   ({a} reads as {direction} confident)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe", type=Path)
    args = ap.parse_args()
    rows = json.loads(args.probe.read_text())
    print(f"{len(rows)} observations over {len({r['seed'] for r in rows})} boards")

    parsed = sum(1 for r in rows if r.get("pass_parsed"))
    print(f"PASS token parsed in {parsed}/{len(rows)} responses "
          f"({100*parsed/max(1,len(rows)):.0f}%)")
    dr = [r["n_decoys_ranked"] for r in rows if r.get("n_decoys_ranked") is not None]
    if dr:
        print(f"decoys included in the ranking: mean {np.mean(dr):.2f} of "
              f"{rows[0].get('n_decoys', 5)}  "
              f"(0 would mean the model simply filters intruders out)")

    # PASS later in the ranking = more words recognised = more confident
    describe(rows, "pass_pos", "PASS position (fraction of the board above it)", True)
    # a decoy surfacing early = the clue looks no better than an unrelated word
    describe(rows, "top_decoy", "best decoy's rank (fraction of list above it)", True)
    describe(rows, "board_below_decoy", "board words ranked BELOW the best decoy", False)

    print("\nA grade ordering that holds with p < 0.05 means the anchor carries signal.")
    print("Flat across grades means it is inert, and a re-collection would buy nothing.")


if __name__ == "__main__":
    main()
