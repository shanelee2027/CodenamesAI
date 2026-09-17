"""Print the board layout + turn-by-turn transcript for games persisted
by codenames/llm_store.py::GameRecordStore (see --record-games on
scripts/pipeline/run_two_team_arena.py) -- the durable, replayable-free
alternative to scripts/tools/scratch_llm_transcripts.py's one-off prints.

The board is shown as the real 5x5 grid, regenerated from the seed rather
than read from the record: `codenames/two_team_arena.py` builds every board
with `Board.generate(seed=seed)` and the default wordlist, so the layout is
fully reproducible, whereas the stored `board` blob is grouped by role and
has lost the original order. The word set is checked against the record, so a
mismatch is reported rather than silently drawn wrong.

**Use --sample, not --seed, to look at "what the model does."** Picking games
to read by eye -- a big win, an assassin death -- hides base rates: a clue
repetition that looks like a quirk in a game chosen for its ending turns out
to happen in 30% of games when the games are drawn at random (see docs/log.md,
2026-09-17). --sample takes a uniform random draw with a fixed seed so the
sample is reproducible and nobody, including the author, gets to choose it.

Usage:
    python scripts/tools/dump_game_records.py cache/llm_store.db
    python scripts/tools/dump_game_records.py cache/llm_store.db --label "learned:noise_0_08+llm"
    python scripts/tools/dump_game_records.py cache/llm_store.db --seed 3
    python scripts/tools/dump_game_records.py cache/llm_store.db \
        --label-prefix 'expected_words[sigma=1.5]' --sample 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, Role
from codenames.llm_store import GameRecordStore

GRID_COLS = 5

# Two-character marks so every cell is the same width and the grid stays aligned.
MARK = {"own": " A", "opponent": " B", "neutral": " .", "assassin": "XX"}
ROLE_KEY = {
    Role.OWN: "own",
    Role.OPPONENT: "opponent",
    Role.NEUTRAL: "neutral",
    Role.ASSASSIN: "assassin",
}


def board_grid(seed: int, recorded: dict[str, list[str]]) -> list[str]:
    """The board as it was laid out, one line per row.

    Roles are printed from the regenerated board, not the record, but the two
    are cross-checked: if the word sets disagree the caller is told instead of
    being shown a plausible-looking board that is not the one that was played.
    """
    board = Board.generate(seed=seed)
    live = {w.lower() for w in board.words}
    stored = {w.lower() for ws in recorded.values() for w in ws}
    if live != stored:
        missing = sorted(stored - live)[:4]
        return [
            f"  (cannot redraw the grid: board for seed {seed} regenerates to a different",
            f"   word set than the record -- e.g. {missing}. Wordlist or seeding changed",
            "   since this game was recorded; falling back to the grouped listing.)",
        ]

    lines = []
    revealed_roles = {c.word: ROLE_KEY[c.role] for c in board.cards}
    words = list(board.words)
    width = max(len(w) for w in words) + 1
    for r in range(0, len(words), GRID_COLS):
        row = words[r : r + GRID_COLS]
        lines.append("  " + "".join(f"{MARK[revealed_roles[w]]} {w:<{width}}" for w in row))
    return lines


def render(row, *, grid: bool = True) -> None:
    recorded = json.loads(row["board"])
    turns = json.loads(row["turns"])
    total_reward = json.loads(row["total_reward"])
    seed = row["seed"]

    print(f"\n{'=' * 78}")
    print(f"seed={seed} label={row['label']!r}")
    print(f"outcome={row['outcome']} winner={row['winner']} total_reward={total_reward}")
    print()
    if grid:
        print("  starting board (A = team A's word, B = team B's, . = neutral, XX = ASSASSIN):")
        for line in board_grid(seed, recorded):
            print(line)
        print()
    print(f"  A ({len(recorded.get('own', []))}): {', '.join(recorded.get('own', []))}")
    print(f"  B ({len(recorded.get('opponent', []))}): {', '.join(recorded.get('opponent', []))}")
    print(f"  .  ({len(recorded.get('neutral', []))}): {', '.join(recorded.get('neutral', []))}")
    print(f"  XX: {', '.join(recorded.get('assassin', []))}")
    print()
    # Which spymaster sat on which side, when the label records it.
    _, _, sides = row["label"].partition("|") if row["label"] else ("", "", "")
    name_of = dict(p.split("=", 1) for p in sides.split(",") if "=" in p) if sides else {}
    if name_of:
        for side in ("A", "B"):
            if side in name_of:
                print(f"  team {side} spymaster: {name_of[side]}")
        print()
    for t in turns:
        who = f"{t['team']}={name_of[t['team']]}" if t["team"] in name_of else t["team"]
        guesses_str = ", ".join(f"{w}({role.upper()})" for w, role in t["guesses"]) or "(none)"
        flag = "  <-- ASSASSIN" if t["ended_reason"] == "assassin" else ""
        print(f"  [{who}] clue={t['clue']!r} n={t['number']} -> {guesses_str}  [{t['ended_reason']}]{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--label", default=None, help="only show games recorded under this exact --run-label")
    parser.add_argument("--label-prefix", default=None,
                        help="only show games whose label starts with this (matches both side assignments)")
    parser.add_argument("--seed", type=int, default=None, help="only show this board seed")
    parser.add_argument("--sample", type=int, default=None,
                        help="show this many games drawn uniformly at random -- the honest way to "
                             "look at behaviour, since choosing games by eye hides base rates")
    parser.add_argument("--sample-seed", type=int, default=0, help="RNG seed for --sample (default 0, reproducible)")
    parser.add_argument("--no-grid", action="store_true", help="skip the 5x5 layout, print only the grouped lists")
    args = parser.parse_args()

    store = GameRecordStore(args.db_path)
    rows = store.all_games(label=args.label)
    if args.label_prefix is not None:
        rows = [r for r in rows if (r["label"] or "").startswith(args.label_prefix)]
    if args.seed is not None:
        rows = [r for r in rows if r["seed"] == args.seed]

    population = len(rows)
    if args.sample is not None and population > args.sample:
        rows = random.Random(args.sample_seed).sample(rows, args.sample)
        print(f"random sample of {len(rows)} from {population} matching games "
              f"(--sample-seed {args.sample_seed}; re-run for the same draw)")

    for row in rows:
        render(row, grid=not args.no_grid)

    print(f"\n{len(rows)} game(s) shown"
          + (f" of {population} matching." if population != len(rows) else "."))
    store.close()


if __name__ == "__main__":
    main()
