"""Run the two-team self-play arena (codenames/two_team_arena.py): the
SAME codemaster+guesser pair on both sides of a real two-team game (see
codenames/game.py::play_two_team_game), across many seeded boards.

Usage:
    python scripts/run_two_team_arena.py --n-boards 300 --codemaster centroid --guesser noisy_glove
    python scripts/run_two_team_arena.py --n-boards 300 \\
        --checkpoint cache/m9/checkpoints/noise_0_08/scorer_best.pt --guesser noisy_glove
    python scripts/run_two_team_arena.py --n-boards 300 \\
        --checkpoint cache/blend_pool/checkpoints/scorer_best.pt \\
        --guesser-pool-config configs/guesser_pool_blend.json --guesser blend

With --checkpoint, routes through codenames/two_team_gpu_arena.py's
batched-across-games GPU path by default (mirrors scripts/run_arena.py's
--gpu-batch-size for the single-team case -- pass --no-gpu-batch for the
normal per-process CPU path instead). A baseline --codemaster always
runs through the normal per-process path either way, since it's already
cheap and has nothing to gain from batching (it scores a handful of
candidates, not the whole clue vocabulary, each turn).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from codenames.codemasters import CentroidCodemaster, LearnedCodemaster, LinearScorerCodemaster, OracleCodemaster, RandomCodemaster
from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.two_team_arena import run_two_team_self_play
from codenames.two_team_gpu_arena import run_two_team_self_play_gpu

BASE_CODEMASTER_SPECS: dict[str, tuple[type, dict]] = {
    "random": (RandomCodemaster, {"seed": 0}),
    "centroid": (CentroidCodemaster, {"seed": 0}),
    "linear_scorer": (LinearScorerCodemaster, {}),
    "oracle": (OracleCodemaster, {}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boards", type=int, default=20, help="number of fixed seeded boards to play (seeds 0..n-1)")
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument("--guesser", required=True, help="name of one guesser in --guesser-pool-config to use on both sides")
    parser.add_argument("--codemaster", choices=list(BASE_CODEMASTER_SPECS), default=None, help="a baseline codemaster")
    parser.add_argument("--checkpoint", type=Path, default=None, help="a learned scorer checkpoint instead of a baseline codemaster")
    parser.add_argument("--risk-aversion", type=float, default=None, help="miss_penalty for a learned codemaster (default: -10.0)")
    parser.add_argument("--max-turns", type=int, default=None, help="override codenames.game.DEFAULT_MAX_TURNS (per team)")
    parser.add_argument("--max-workers", type=int, default=None, help="default: os.cpu_count() -- only used without --checkpoint's GPU path")
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=32,
        help="with --checkpoint, batch of simultaneous two-team games scored per forward pass "
        "(codenames/two_team_gpu_arena.py). Falls back to CPU automatically if no CUDA device is available.",
    )
    parser.add_argument("--no-gpu-batch", action="store_true", help="use the normal per-process path for --checkpoint too, instead of --gpu-batch-size")
    parser.add_argument("--sims-cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="only used by the GPU-batched --checkpoint path")
    args = parser.parse_args()

    if (args.codemaster is None) == (args.checkpoint is None):
        parser.error("pass exactly one of --codemaster or --checkpoint")

    use_gpu_batch = args.checkpoint is not None and not args.no_gpu_batch

    if args.checkpoint is not None:
        codemaster_cls, codemaster_kwargs = LearnedCodemaster, {"checkpoint_path": args.checkpoint}
        if args.risk_aversion is not None:
            codemaster_kwargs["miss_penalty"] = args.risk_aversion
        codemaster_label = f"learned:{args.checkpoint.parent.name}"
    else:
        codemaster_cls, codemaster_kwargs = BASE_CODEMASTER_SPECS[args.codemaster]
        codemaster_label = args.codemaster

    kwargs = {}
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns

    seeds = list(range(args.n_boards))
    start = time.time()
    if use_gpu_batch:
        sims = SimilarityTensor.load(args.sims_cache_dir)
        learned_codemaster = codemaster_cls(**codemaster_kwargs)
        result = run_two_team_self_play_gpu(
            codemaster=learned_codemaster,
            guesser_pool_config=args.guesser_pool_config,
            guesser_name=args.guesser,
            seeds=seeds,
            sims=sims,
            batch_size=args.gpu_batch_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        )
    else:
        result = run_two_team_self_play(
            codemaster_cls,
            codemaster_kwargs,
            args.guesser_pool_config,
            args.guesser,
            seeds,
            max_workers=args.max_workers,
            **kwargs,
        )
    elapsed = time.time() - start

    print(f"{result.n_games} two-team games ({codemaster_label} + {args.guesser} on both sides) in {elapsed:.1f}s\n")
    print(f"{'assassin-hit rate':22s} {100 * result.assassin_rate:6.1f}%")
    print(f"{'half-turns (all)':22s} {result.mean_half_turns_all:6.2f}")
    turns_clean = f"{result.mean_half_turns_clean_finish:.2f}" if result.mean_half_turns_clean_finish is not None else "--"
    print(f"{'half-turns (clean finish)':22s} {turns_clean:>6s}")
    print()
    print("per-guess role breakdown (of every word actually guessed, pooled across both teams):")
    print(
        f"  own {100 * result.guess_own_rate:5.1f}%   opponent {100 * result.guess_opponent_rate:5.1f}%   "
        f"neutral {100 * result.guess_neutral_rate:5.1f}%   assassin {100 * result.guess_assassin_rate:5.1f}%"
    )


if __name__ == "__main__":
    main()
