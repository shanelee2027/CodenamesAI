"""Run the frozen eval suite (configs/eval_suite.json): a challenger against
an opponent on the held-out boards, every board played both ways, against
the suite's fixed LLM guesser. This SPENDS MONEY on the first run of a pair;
reruns and extensions pay only for boards not yet recorded.

    python scripts/pipeline/run_eval_suite.py learned_listener expected_words \\
        --opponent-param sigma=1.5 --dry-run
    python scripts/pipeline/run_eval_suite.py learned_listener expected_words \\
        --opponent-param sigma=1.5 --max-workers 6 --threads-per-worker 16

Spymasters are registry names (configs/spymasters.json); `--*-param k=v`
overrides a constructor parameter. Every parameter and model file goes into
the spymaster's identity (codenames/eval_suite.py::spymaster_identity), so a
changed setting is a different model in the store, never a reuse.

The headline is the sign test on boards one side won both ways -- the
paired comparison. Game win rate is reported beside it for reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from codenames.env import load_env
from codenames.eval_suite import DEFAULT_EVAL_SUITE_CONFIG, load_eval_suite, missing_seeds, run_eval_suite, spymaster_identity
from codenames.llm_store import DEFAULT_DB_PATH
from codenames.spymasters.registry import spymaster_spec
from codenames.stats import fisher_2x2


def parse_value(v: str):
    """None, then bool, then int, then float, then string."""
    if v.lower() == "none":
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def resolve(name: str, overrides: list[str]) -> tuple[tuple, str]:
    params = {}
    for kv in overrides:
        k, _, v = kv.partition("=")
        params[k] = parse_value(v)
    cls, kwargs = spymaster_spec(name, **params)
    return (cls, kwargs), spymaster_identity(name, kwargs, cls.model_files(kwargs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("challenger")
    ap.add_argument("opponent")
    ap.add_argument("--challenger-param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--opponent-param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--suite", type=Path, default=DEFAULT_EVAL_SUITE_CONFIG)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--threads-per-worker", type=int, default=16,
                    help="see codenames/two_team_arena.py::run_two_team_matchup")
    ap.add_argument("--dry-run", action="store_true", help="say what would be played, then stop")
    ap.add_argument("--boards", type=int, default=None,
                    help="play only the suite's first N boards -- a pilot. suite_id does not "
                         "depend on the board list, so a later full run plays just the rest")
    args = ap.parse_args()

    suite = load_eval_suite(args.suite)
    if args.boards is not None:
        suite = dataclasses.replace(suite, board_seeds=tuple(suite.board_seeds[:args.boards]))
    spec_x, id_x = resolve(args.challenger, args.challenger_param)
    spec_y, id_y = resolve(args.opponent, args.opponent_param)
    if id_x == id_y:
        raise SystemExit(f"challenger and opponent are the same model ({id_x})")
    ids = (id_x, id_y)
    todo = missing_seeds(ids, suite, args.db)
    print(f"suite {suite.name} ({suite.suite_id})  guesser {suite.guesser}  {len(suite.board_seeds)} boards")
    print(f"  challenger {id_x}  {spec_x[1]}")
    print(f"  opponent   {id_y}  {spec_y[1]}")
    print(f"  {len(todo)} board(s) to play, {2 * len(todo)} games")
    if args.dry_run:
        return

    load_env()
    t0 = time.time()
    res = run_eval_suite(spec_x, spec_y, ids, suite, game_record_db=args.db,
                         max_workers=args.max_workers, threads_per_worker=args.threads_per_worker,
                         progress=True)
    s = res.paired
    print(f"\nplayed {res.played} board(s) in {time.time() - t0:.0f}s"
          + (f"; {len(res.discarded)} discarded (guesser refused), retried next run" if res.discarded else ""))
    print(f"{s.boards} boards played both ways, {s.games} games\n")
    print(f"{'':28s} {'win%':>6s} {'95% CI':>14s} {'swept':>6s} {'assassin':>9s} {'mean k':>7s} {'own/clue':>9s} {'own%':>6s}")
    for i in ids:
        lo, hi = s.wilson(i)
        t = res.turns.get(i)
        print(f"{i:28s} {s.win_rate(i):6.1%}  [{lo:.2f}, {hi:.2f}] {s.swept.get(i, 0):6d} "
              f"{s.assassin.get(i, 0):9d} {t.mean_k if t else 0:7.2f} {t.own_per_clue if t else 0:9.2f} "
              f"{t.own_rate if t else 0:6.1%}")
    if s.decisive():
        print(f"\nsign test on {s.decisive()} decisive boards (split {s.split}): p = {s.sign_p(id_x):.4f}")
    ax, ay = s.assassin.get(id_x, 0), s.assassin.get(id_y, 0)
    print(f"assassin losses {ax} vs {ay}: Fisher p = {fisher_2x2(ax, s.games - ax, ay, s.games - ay):.3f}")


if __name__ == "__main__":
    main()
