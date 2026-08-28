"""Run the cross-play arena (SCOPE.md §M6): every codemaster x every
guesser (held-out included -- see codenames/arena.py's module docstring for
why), over a fixed set of seeded boards. Prints the win-rate / assassin-rate
/ mean-turns / mean-own-words-per-clue matrix and per-worker peak RSS.

Usage:
    python scripts/run_arena.py --n-boards 20 --max-workers 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from codenames.arena import run_arena
from codenames.codemasters import CentroidCodemaster, LearnedCodemaster, LinearScorerCodemaster, RandomCodemaster
from codenames.guessers.registry import DEFAULT_POOL_CONFIG

BASE_CODEMASTER_SPECS = {
    "random": (RandomCodemaster, {"seed": 0}),
    "centroid": (CentroidCodemaster, {"seed": 0}),
    "linear_scorer": (LinearScorerCodemaster, {}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boards", type=int, default=20, help="number of fixed seeded boards to play (seeds 0..n-1)")
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument("--db", type=Path, default=Path("cache/arena.db"))
    parser.add_argument("--max-turns", type=int, default=None, help="override codenames.game.DEFAULT_MAX_TURNS")
    parser.add_argument("--max-workers", type=int, default=None, help="default: os.cpu_count()")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="scorer checkpoint from scripts/train_scorer.py -- adds a 'learned' codemaster if given"
    )
    parser.add_argument("--risk-aversion", type=float, default=None, help="miss_penalty for the learned codemaster (default: -10.0, see codenames.scorer)")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.n_boards))

    codemaster_specs = dict(BASE_CODEMASTER_SPECS)
    if args.checkpoint is not None:
        learned_kwargs = {"checkpoint_path": args.checkpoint}
        if args.risk_aversion is not None:
            learned_kwargs["miss_penalty"] = args.risk_aversion
        codemaster_specs["learned"] = (LearnedCodemaster, learned_kwargs)

    kwargs = {}
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns

    start = time.time()
    results, worker_rss = run_arena(
        codemaster_specs=codemaster_specs,
        guesser_pool_config=args.guesser_pool_config,
        seeds=seeds,
        db_path=args.db,
        max_workers=args.max_workers,
        **kwargs,
    )
    elapsed = time.time() - start

    print(f"{len(results)} codemaster x guesser pairs, {len(seeds)} boards each, in {elapsed:.1f}s")
    print(f"logged to {args.db}\n")

    header = f"{'codemaster':16s} {'guesser':20s} {'held-out':9s} {'win%':>7s} {'assassin%':>10s} {'turns':>7s} {'own/clue':>9s}"
    print(header)
    print("-" * len(header))
    for (cm_name, g_name), r in sorted(results.items()):
        print(
            f"{cm_name:16s} {g_name:20s} {'yes' if r.held_out else 'no':9s} "
            f"{100 * r.win_rate:6.1f}% {100 * r.assassin_rate:9.1f}% {r.mean_turns:7.2f} {r.mean_own_words_per_clue:9.3f}"
        )

    print(f"\nper-worker peak RSS ({len(worker_rss)} worker process(es)):")
    for pid, rss_kb in sorted(worker_rss.items()):
        print(f"  pid {pid}: {rss_kb / 1024:.1f} MB")


if __name__ == "__main__":
    main()
