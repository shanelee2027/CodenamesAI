"""Board-level statistics for a head-to-head arena run.

`run_two_team_arena.py` plays every board twice with the sides swapped, so its
40 games are 20 matched pairs, not 40 independent draws. Treating them as
independent both overstates the sample and ignores the pairing that the swap
exists to exploit. The right unit is the board: within a pair the two games
share a board layout, so a board where each side wins once carries no evidence
about the spymasters and a board swept 2-0 carries the most.

Reported per matchup:
  * the sweep table -- how many boards each side took 2-0, and how many split
  * an exact binomial sign test on the swept boards, which is the paired test
  * a Wilson interval on the game win rate over boards played both ways
  * assassin counts, by Fisher's exact test

Usage:
    python scripts/tools/analyze_headtohead.py cache/llm_store.db --contains learned_listener
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

from codenames.headtohead import paired_summary, seat_names
from codenames.stats import fisher_2x2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path)
    ap.add_argument("--contains", default="", help="only labels containing this string")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("select label, seed, winner, turns from game_records where label like '%|A=%'").fetchall()
    con.close()

    runs: dict[str, list] = defaultdict(list)
    for row in rows:
        if args.contains and args.contains not in row[0]:
            continue
        runs[row[0].partition("|")[0]].append(row)

    for run, run_rows in sorted(runs.items()):
        names = sorted({n for r in run_rows for n in seat_names(r[0])})
        if len(names) != 2:
            print(f"{run}: expected 2 spymasters, saw {names}")
            continue
        x, y = names
        s = paired_summary(run_rows)
        print(f"\n=== {run} ===")
        print(f"  boards played both ways {s.boards}   games {s.games}")
        print(f"  {'spymaster':30s} {'games won':>10s} {'win%':>7s} {'95% CI':>16s} {'assassin':>9s}")
        for nm in (x, y):
            lo, hi = s.wilson(nm)
            print(f"  {nm:30s} {s.wins.get(nm, 0):10d} {s.win_rate(nm):6.1%} "
                  f"  [{lo:.2f}, {hi:.2f}] {s.assassin.get(nm, 0):9d}")
        print(f"  paired boards: {x} swept {s.swept.get(x, 0)}, {y} swept {s.swept.get(y, 0)}, split {s.split}")
        if s.decisive():
            p = s.sign_p(y)
            verdict = "significant" if p < 0.05 else "NOT significant"
            print(f"  sign test on {s.decisive()} decisive boards: p = {p:.4f}  ({verdict} at 0.05)")
        else:
            print("  every board split -- no paired evidence either way")
        ax, ay = s.assassin.get(x, 0), s.assassin.get(y, 0)
        pa = fisher_2x2(ax, s.games - ax, ay, s.games - ay)
        print(f"  assassin {x} {ax} vs {y} {ay}: Fisher p = {pa:.3f}")


if __name__ == "__main__":
    main()
