"""Run the cross-play arena: every spymaster x every
guesser (held-out included -- see codenames/arena.py's module docstring for
why), over a fixed set of seeded boards. Prints the win-rate / assassin-rate
/ mean-turns / mean-own-words-per-clue matrix and per-worker peak RSS.

Usage:
    python scripts/pipeline/run_arena.py --n-boards 20 --max-workers 8
    python scripts/pipeline/run_arena.py --n-boards 300 --max-workers 8

Every spymaster with role "baseline" in configs/spymasters.json runs here,
selected by role rather than by name so a new entry needs no edit to this
file. Per docs/design-decisions.md, this matrix is a *training
diagnostic*, not a scoreboard: its guessers are the synthetic training
pool, and evaluation results come only from the frozen LLM eval suite
(codenames/eval_suite.py).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from codenames.arena import run_arena
from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.spymasters.registry import load_spymasters, spymaster_names

# This script's spymasters, selected by role from configs/spymasters.json
# rather than by name, so a new entry needs no edit here. Entries with the
# "exploration" role are offered only by scripts/pipeline/run_two_team_arena.py.
BASE_SPYMASTER_NAMES = spymaster_names("baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boards", type=int, default=20, help="number of fixed seeded boards to play (seeds 0..n-1)")
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument("--db", type=Path, default=Path("cache/arena.db"))
    parser.add_argument("--max-turns", type=int, default=None, help="override codenames.game.DEFAULT_MAX_TURNS")
    parser.add_argument("--max-workers", type=int, default=None, help="default: os.cpu_count()")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.n_boards))

    all_entries = load_spymasters()
    spymaster_specs = {name: all_entries[name].spec for name in BASE_SPYMASTER_NAMES}

    kwargs = {}
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns

    start = time.time()
    results, worker_rss = run_arena(
        spymaster_specs=spymaster_specs,
        guesser_pool_config=args.guesser_pool_config,
        seeds=seeds,
        db_path=args.db,
        max_workers=args.max_workers,
        **kwargs,
    )

    elapsed = time.time() - start

    print(f"{len(results)} spymaster x guesser pairs, {len(seeds)} boards each, in {elapsed:.1f}s")
    print(f"logged to {args.db}\n")

    header = (
        f"{'spymaster':16s} {'guesser':20s} {'held-out':9s} {'win%':>7s} {'assassin%':>10s} "
        f"{'turns(all)':>11s} {'turns(win)':>11s} {'own/clue':>9s}"
    )
    print(header)
    print("-" * len(header))
    for (cm_name, g_name), r in sorted(results.items()):
        turns_on_win = f"{r.mean_turns_on_win:11.2f}" if r.mean_turns_on_win is not None else f"{'--':>11s}"
        print(
            f"{cm_name:16s} {g_name:20s} {'yes' if r.held_out else 'no':9s} "
            f"{100 * r.win_rate:6.1f}% {100 * r.assassin_rate:9.1f}% {r.mean_turns:11.2f} {turns_on_win} {r.mean_own_words_per_clue:9.3f}"
        )

    print("\nper-guess role breakdown (of every word actually guessed, not per-game):")
    breakdown_header = f"{'spymaster':16s} {'guesser':20s} {'own%':>7s} {'opponent%':>10s} {'neutral%':>9s} {'assassin%':>10s}"
    print(breakdown_header)
    print("-" * len(breakdown_header))
    for (cm_name, g_name), r in sorted(results.items()):
        print(
            f"{cm_name:16s} {g_name:20s} "
            f"{100 * r.guess_own_rate:6.1f}% {100 * r.guess_opponent_rate:9.1f}% "
            f"{100 * r.guess_neutral_rate:8.1f}% {100 * r.guess_assassin_rate:9.1f}%"
        )

    print(f"\nper-worker peak RSS ({len(worker_rss)} worker process(es)):")
    for pid, rss_kb in sorted(worker_rss.items()):
        print(f"  pid {pid}: {rss_kb / 1024:.1f} MB")


if __name__ == "__main__":
    main()
