"""Print the board layout + turn-by-turn transcript for games persisted
by codenames/llm_store.py::GameRecordStore (see --record-games on
scripts/run_two_team_arena.py) -- the durable, replayable-free
alternative to scripts/scratch_llm_transcripts.py's one-off prints.

Usage:
    python scripts/dump_game_records.py cache/llm_store.db
    python scripts/dump_game_records.py cache/llm_store.db --label "learned:noise_0_08+llm"
    python scripts/dump_game_records.py cache/llm_store.db --seed 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codenames.llm_store import GameRecordStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--label", default=None, help="only show games recorded under this --run-label")
    parser.add_argument("--seed", type=int, default=None, help="only show this board seed")
    args = parser.parse_args()

    store = GameRecordStore(args.db_path)
    rows = store.all_games(label=args.label)
    if args.seed is not None:
        rows = [r for r in rows if r["seed"] == args.seed]

    for row in rows:
        board = json.loads(row["board"])
        turns = json.loads(row["turns"])
        total_reward = json.loads(row["total_reward"])

        print(f"\n{'=' * 70}")
        print(f"seed={row['seed']} label={row['label']!r} outcome={row['outcome']} winner={row['winner']} total_reward={total_reward}")
        print(f"  Team A's own:  {', '.join(board.get('own', []))}")
        print(f"  Team B's own:  {', '.join(board.get('opponent', []))}")
        print(f"  Neutral:       {', '.join(board.get('neutral', []))}")
        print(f"  Assassin:      {', '.join(board.get('assassin', []))}")
        print()
        for t in turns:
            guesses_str = ", ".join(f"{w}({role.upper()})" for w, role in t["guesses"])
            flag = "  <-- ASSASSIN" if t["ended_reason"] == "assassin" else ""
            print(f"  [{t['team']}] clue={t['clue']!r} n={t['number']} -> {guesses_str}  [{t['ended_reason']}]{flag}")

    print(f"\n{len(rows)} game(s) shown.")
    store.close()


if __name__ == "__main__":
    main()
