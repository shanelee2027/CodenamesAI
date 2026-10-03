"""Play many games with a simulated guesser, for training a board-reading
value model (docs/log.md, "Simulated games").

The guesser is a fitted listener sampling rankings
(codenames/guessers/listener_sample.py), so a game costs compute, not money.
Boards are generated from the training word list on their own seed range
(6,000,000 on), never the eval suites' boards or words. Every game is
recorded into a separate store, `cache/sim_games.db`, never the LLM store,
with the label `<label>|A=<name>,B=<name>`, so
scripts/tools/eval_board_value.py's state loader reads it unchanged.

Boards are played in chunks, both seatings each, and a chunk already in the
store is skipped, so the run can be stopped and resumed.

    python scripts/pipeline/simulate_games.py --label sim_v1 --boards 5000 \\
        --challenger win_prob_listener --challenger-param turn_model=frozen \\
        --challenger-param model_path=cache/listener_gbt.txt --opponent learned_listener
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

SIM_DB = PROJECT_ROOT / "cache" / "sim_games.db"
SEED_BASE = 6_000_000


def done_seeds(db: Path, label: str) -> set[int]:
    if not db.exists():
        return set()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT seed, count(*) FROM game_records WHERE label LIKE ? GROUP BY seed",
                            (label + "|%",)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {s for s, n in rows if n >= 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--boards", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0, help="board offset: seeds SEED_BASE + start + i")
    ap.add_argument("--guesser", default="sim:assoc")
    ap.add_argument("--challenger", default="win_prob_listener")
    ap.add_argument("--challenger-param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--opponent", default="learned_listener")
    ap.add_argument("--opponent-param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--chunk", type=int, default=200, help="boards per chunk")
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--db", type=Path, default=SIM_DB)
    args = ap.parse_args()

    from codenames.board import load_training_wordlist
    from codenames.two_team_arena import run_two_team_matchup
    from run_eval_suite import resolve

    spec_x, name_x = resolve(args.challenger, args.challenger_param)
    spec_y, name_y = resolve(args.opponent, args.opponent_param)
    vocab = load_training_wordlist()
    seeds = [SEED_BASE + args.start + i for i in range(args.boards)]
    have = done_seeds(args.db, args.label)
    todo = [s for s in seeds if s not in have]
    print(f"{args.label}: {name_x} vs {name_y}, guesser {args.guesser}; {len(todo)} of {len(seeds)} boards to play",
          flush=True)
    t0 = time.time()
    played = 0
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        res = run_two_team_matchup(spec_x, spec_y, (name_x, name_y), args.guesser, chunk,
                                   game_record_db=args.db, run_label=args.label, max_workers=args.max_workers,
                                   threads_per_worker=1, vocabulary=vocab)
        played += len(chunk)
        rate = played / (time.time() - t0)
        x = res.sides.get(name_x) if hasattr(res, "sides") else None
        print(f"  {played}/{len(todo)} boards  {rate * 3600:.0f} boards/h"
              + (f"  {name_x} win {x.win_rate:.3f}" if x is not None else ""), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
