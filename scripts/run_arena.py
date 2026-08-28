"""Run the cross-play arena (SCOPE.md §M6): every codemaster x every
guesser (held-out included -- see codenames/arena.py's module docstring for
why), over a fixed set of seeded boards. Prints the win-rate / assassin-rate
/ mean-turns / mean-own-words-per-clue matrix and per-worker peak RSS.

Usage:
    python scripts/run_arena.py --n-boards 20 --max-workers 8
    python scripts/run_arena.py --n-boards 300 --checkpoint cache/m9/checkpoints/noise_0_08/scorer_best.pt

With --checkpoint, the learned codemaster routes through
codenames/gpu_arena.py's batched-across-games GPU path by default (13x
faster measured in practice, see docs/log.md -- pass --no-gpu-batch for
the normal per-process CPU path instead). Baselines
(random/centroid/linear_scorer) always run through the regular arena
either way, since they're already cheap and have nothing to gain from
batching. Results are merged into one report. See
codenames/gpu_arena.py's module docstring for why this exists and what it
does and doesn't accelerate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from codenames.arena import run_arena
from codenames.codemasters import CentroidCodemaster, LearnedCodemaster, LinearScorerCodemaster, RandomCodemaster
from codenames.gpu_arena import run_arena_gpu
from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

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
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=32,
        help="with --checkpoint, run the learned codemaster through codenames/gpu_arena.py's GPU-batched "
        "path (default: on, batch of 32) instead of the normal per-process one -- measured 13x faster in "
        "practice (docs/log.md). Baselines still run through the normal path either way. Falls back to "
        "CPU automatically if no CUDA device is available, just without the speedup.",
    )
    parser.add_argument(
        "--no-gpu-batch", action="store_true", help="use the normal per-process path for the learned codemaster too, instead of --gpu-batch-size"
    )
    parser.add_argument("--sims-cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="only used by --gpu-batch-size")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.n_boards))
    use_gpu_batch = args.checkpoint is not None and not args.no_gpu_batch

    codemaster_specs = dict(BASE_CODEMASTER_SPECS)
    learned_kwargs = {}
    if args.checkpoint is not None:
        learned_kwargs = {"checkpoint_path": args.checkpoint}
        if args.risk_aversion is not None:
            learned_kwargs["miss_penalty"] = args.risk_aversion
        if not use_gpu_batch:
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

    if use_gpu_batch:
        sims = SimilarityTensor.load(args.sims_cache_dir)
        learned_codemaster = LearnedCodemaster(**learned_kwargs)
        gpu_results = run_arena_gpu(
            codemaster=learned_codemaster,
            codemaster_name="learned",
            guesser_pool_config=args.guesser_pool_config,
            seeds=seeds,
            db_path=args.db,
            sims=sims,
            batch_size=args.gpu_batch_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        )
        results.update({("learned", g_name): r for g_name, r in gpu_results.items()})

    elapsed = time.time() - start

    print(f"{len(results)} codemaster x guesser pairs, {len(seeds)} boards each, in {elapsed:.1f}s")
    print(f"logged to {args.db}\n")

    header = (
        f"{'codemaster':16s} {'guesser':20s} {'held-out':9s} {'win%':>7s} {'assassin%':>10s} "
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
    breakdown_header = f"{'codemaster':16s} {'guesser':20s} {'own%':>7s} {'opponent%':>10s} {'neutral%':>9s} {'assassin%':>10s}"
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
