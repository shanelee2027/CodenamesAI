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
  * a Wilson interval on the raw game win rate, for comparability with the
    arena's own printout
  * assassin counts, by Fisher's exact test

Usage:
    python scripts/tools/analyze_headtohead.py cache/llm_store.db --contains learned_listener
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p = 0.5."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * 0.5 ** n
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a, b], [c, d]]."""
    n = a + b + c + d
    def p(x):
        return (math.comb(a + b, x) * math.comb(c + d, a + c - x)) / math.comb(n, a + c)
    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path)
    ap.add_argument("--contains", default="", help="only labels containing this string")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    rows = con.execute("select label, seed, winner, turns from game_records").fetchall()

    runs: dict[str, dict[int, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for label, seed, winner, turns in rows:
        if args.contains and args.contains not in label:
            continue
        run, _, sides = label.partition("|")
        a, _, b = sides.partition(",")
        names = (a.split("=", 1)[1], b.split("=", 1)[1])
        # `winner` is a side, 'A' or 'B'; map it to whichever spymaster sat there.
        won = names[0] if str(winner).upper().endswith("A") else names[1]
        # `outcome` is only win/loss, so who walked into the assassin has to
        # come from the turn log: the team whose own turn ended that way.
        killer = None
        for t in json.loads(turns):
            if t.get("ended_reason") == "assassin":
                killer = names[0] if str(t.get("team", "")).upper() == "A" else names[1]
        runs[run][seed].append((won, killer))

    for run, boards in sorted(runs.items()):
        names = sorted({w for gs in boards.values() for w, _ in gs})
        if len(names) != 2:
            print(f"{run}: expected 2 spymasters, saw {names}")
            continue
        x, y = names
        sweep_x = sweep_y = split = 0
        wins = {x: 0, y: 0}
        assassin = {x: 0, y: 0}
        for seed, gs in boards.items():
            for w, killer in gs:
                wins[w] += 1
                if killer is not None:
                    assassin[killer] += 1
            if len(gs) == 2:
                w0, w1 = gs[0][0], gs[1][0]
                if w0 == w1 == x:
                    sweep_x += 1
                elif w0 == w1 == y:
                    sweep_y += 1
                else:
                    split += 1

        n_games = sum(wins.values())
        decisive = sweep_x + sweep_y
        print(f"\n=== {run} ===")
        print(f"  boards {len(boards)}   games {n_games}")
        print(f"  {'spymaster':30s} {'games won':>10s} {'win%':>7s} {'95% CI':>16s} {'assassin':>9s}")
        for nm in (x, y):
            lo, hi = wilson(wins[nm], n_games)
            print(f"  {nm:30s} {wins[nm]:10d} {wins[nm]/n_games:6.1%} "
                  f"  [{lo:.2f}, {hi:.2f}] {assassin[nm]:9d}")
        print(f"  paired boards: {x} swept {sweep_x}, {y} swept {sweep_y}, split {split}")
        if decisive:
            p = binom_two_sided(sweep_y, decisive)
            verdict = "significant" if p < 0.05 else "NOT significant"
            print(f"  sign test on {decisive} decisive boards: p = {p:.4f}  ({verdict} at 0.05)")
        else:
            print("  every board split -- no paired evidence either way")
        pa = fisher_2x2(assassin[x], n_games - assassin[x], assassin[y], n_games - assassin[y])
        print(f"  assassin {x} {assassin[x]} vs {y} {assassin[y]}: Fisher p = {pa:.3f}")


if __name__ == "__main__":
    main()
