"""Two challengers against the same opponent on the same simulated boards,
compared board by board (docs/log.md, "Simulated games").

Each label's games are in cache/sim_games.db (scripts/pipeline/simulate_games.py),
both seatings per board. On the boards both labels played, the challenger's
games won per board (0, 1 or 2) are paired. The guesser's draws are seeded by
(seed, clue, candidates, number), so where the two runs reach the same
position with the same clue they share the guess. Only the challengers'
differences change the games.

Reported: win rates on the shared boards, the paired difference with a
bootstrap over boards, boards where one challenger won more games than the
other (sign test), and assassin losses.

    python scripts/tools/compare_sim_runs.py sim_v1 sim_bv1

Recorded gpt-oss suite games work the same way, with the challengers told
apart by name:

    python scripts/tools/compare_sim_runs.py eval:holdout_v1_gptoss eval:holdout_v1_gptoss \
        --db cache/llm_store.db --base-name win_prob_listener:9c304cc348d6 \
        --other-name board_value_listener:9bd8d05c6da9
"""

from __future__ import annotations

import argparse
import sqlite3
from math import comb
from pathlib import Path

import numpy as np

SIM_DB = Path(__file__).resolve().parents[2] / "cache" / "sim_games.db"


def per_board(db: Path, label: str, opponent: str, challenger: str | None = None) -> tuple[dict[int, list[int]], str]:
    """{seed: [games won, assassin losses, games]} for the side that is not
    `opponent` (only runs where that side's name starts with `challenger`,
    when given), and that side's name."""
    out: dict[int, list[int]] = {}
    name = None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for seed, run, outcome, winner in conn.execute(
            "SELECT seed, label, outcome, winner FROM game_records WHERE label LIKE ?", (label + "|%",)):
        seat = dict(kv.split("=", 1) for kv in run.split("|", 1)[1].split(","))
        if sum(n.startswith(opponent) for n in seat.values()) != 1:
            continue                                                     # not a game against `opponent`
        me = next(s for s, n in seat.items() if not n.startswith(opponent))
        if challenger and not seat[me].startswith(challenger):
            continue
        name = seat[me]
        row = out.setdefault(seed, [0, 0, 0])
        row[0] += winner == me
        row[1] += outcome == "loss" and winner != me                    # we revealed the assassin
        row[2] += 1
    conn.close()
    return {s: r for s, r in out.items() if r[2] == 2}, name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--opponent", default="learned_listener")
    ap.add_argument("--db", type=Path, default=SIM_DB)
    ap.add_argument("--base-name", default=None, help="challenger name prefix within `base`'s runs")
    ap.add_argument("--other-name", default=None, help="challenger name prefix within `other`'s runs")
    args = ap.parse_args()

    A, name_a = per_board(args.db, args.base, args.opponent, args.base_name)
    B, name_b = per_board(args.db, args.other, args.opponent, args.other_name)
    seeds = sorted(set(A) & set(B))
    wa = np.array([A[s][0] for s in seeds], dtype=float)
    wb = np.array([B[s][0] for s in seeds], dtype=float)
    d = wb - wa
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1) / 2
    up, down = int((d > 0).sum()), int((d < 0).sum())
    n = up + down
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
    print(f"{len(seeds)} boards played by both, 2 games each, against {args.opponent}")
    print(f"  {args.base:10s} {name_a}: win {wa.mean() / 2:.3f}, assassin losses {sum(A[s][1] for s in seeds)}")
    print(f"  {args.other:10s} {name_b}: win {wb.mean() / 2:.3f}, assassin losses {sum(B[s][1] for s in seeds)}")
    print(f"  difference {d.mean() / 2:+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] "
          f"per game; boards better {up}, worse {down}, sign p = {p:.3f}; "
          f"identical results on {int((d == 0).sum())} boards")


if __name__ == "__main__":
    main()
