"""Run the two-team self-play arena (codenames/two_team_arena.py): the
SAME spymaster+guesser pair on both sides of a real two-team game (see
codenames/game.py::play_two_team_game), across many seeded boards.

Usage:
    python scripts/run_two_team_arena.py --n-boards 300 --spymaster centroid --guesser noisy_glove
    python scripts/run_two_team_arena.py --n-boards 300 \\
        --checkpoint cache/m9/checkpoints/noise_0_08/scorer_best.pt --guesser noisy_glove
    python scripts/run_two_team_arena.py --n-boards 300 \\
        --checkpoint cache/blend_pool/checkpoints/scorer_best.pt \\
        --guesser-pool-config configs/guesser_pool_blend.json --guesser blend

With --checkpoint, routes through codenames/two_team_gpu_arena.py's
batched-across-games GPU path by default (mirrors scripts/run_arena.py's
--gpu-batch-size for the single-team case -- pass --no-gpu-batch for the
normal per-process CPU path instead). A baseline --spymaster always
runs through the normal per-process path either way, since it's already
cheap and has nothing to gain from batching (it scores a handful of
candidates, not the whole clue vocabulary, each turn).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.registry import load_spymasters, spymaster_names, spymaster_spec
from codenames.two_team_arena import run_two_team_self_play
from codenames.two_team_gpu_arena import run_two_team_self_play_gpu

# This script's baseline set (unchanged from before the registry existed).
# Names into configs/spymasters.json; "learned" is built separately since
# it needs a --checkpoint, supplied per invocation rather than fixed in
# that config.
# Selected by role from configs/spymasters.json rather than by name, so a
# new entry needs no edit here. This script offers the "exploration"
# role (oracle) on top of the standard baselines; scripts/run_arena.py
# does not -- that difference is the only reason the two lists differ.
BASE_SPYMASTER_NAMES = spymaster_names("baseline", "exploration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boards", type=int, default=20, help="number of fixed seeded boards to play (seeds 0..n-1)")
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument(
        "--guesser",
        required=True,
        help="name of one guesser in --guesser-pool-config to use on both sides, or 'mixed' -- each game "
        "independently draws a guesser uniformly from the whole pool, matching the distribution the "
        "spymaster was actually trained against (see codenames/two_team_arena.py::MIXED_GUESSER)",
    )
    parser.add_argument("--spymaster", choices=BASE_SPYMASTER_NAMES, default=None, help="a baseline spymaster")
    parser.add_argument("--checkpoint", type=Path, default=None, help="a learned scorer checkpoint instead of a baseline spymaster")
    parser.add_argument("--risk-aversion", type=float, default=None, help="miss_penalty for a learned spymaster (default: -10.0)")
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
    parser.add_argument(
        "--record-games",
        type=Path,
        default=None,
        help="persist every game's board + turn sequence to this SQLite file (codenames/llm_store.py), "
        "so a run can be inspected later without replaying it -- see scripts/dump_game_records.py. "
        "Most useful when --guesser costs real money per turn (e.g. 'llm').",
    )
    parser.add_argument("--run-label", default=None, help="label stored alongside --record-games' rows (default: '<spymaster>+<guesser>')")
    args = parser.parse_args()

    if (args.spymaster is None) == (args.checkpoint is None):
        parser.error("pass exactly one of --spymaster or --checkpoint")

    use_gpu_batch = args.checkpoint is not None and not args.no_gpu_batch

    if args.checkpoint is not None:
        overrides = {"checkpoint_path": args.checkpoint}
        if args.risk_aversion is not None:
            overrides["miss_penalty"] = args.risk_aversion
        spymaster_cls, spymaster_kwargs = spymaster_spec("learned", **overrides)
        spymaster_label = f"learned:{args.checkpoint.parent.name}"
    else:
        spymaster_cls, spymaster_kwargs = load_spymasters()[args.spymaster].spec
        spymaster_label = args.spymaster

    kwargs = {}
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    run_label = args.run_label if args.run_label is not None else f"{spymaster_label}+{args.guesser}"
    if args.record_games is not None:
        kwargs["game_record_db"] = args.record_games
        kwargs["run_label"] = run_label

    seeds = list(range(args.n_boards))
    start = time.time()
    if use_gpu_batch:
        sims = SimilarityTensor.load(args.sims_cache_dir)
        learned_spymaster = spymaster_cls(**spymaster_kwargs)
        result = run_two_team_self_play_gpu(
            spymaster=learned_spymaster,
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
            spymaster_cls,
            spymaster_kwargs,
            args.guesser_pool_config,
            args.guesser,
            seeds,
            max_workers=args.max_workers,
            **kwargs,
        )
    elapsed = time.time() - start

    print(f"{result.n_games} two-team games ({spymaster_label} + {args.guesser} on both sides) in {elapsed:.1f}s\n")
    print(f"{'assassin-hit rate':22s} {100 * result.assassin_rate:6.1f}%")
    print(f"{'half-turns (all)':22s} {result.mean_half_turns_all:6.2f}")
    turns_clean = f"{result.mean_half_turns_clean_finish:.2f}" if result.mean_half_turns_clean_finish is not None else "--"
    print(f"{'half-turns (clean finish)':22s} {turns_clean:>6s}")
    print(f"{'mean clue number':22s} {result.mean_clue_number:6.2f}")
    print(f"{'mean correct per clue':22s} {result.mean_correct_per_clue:6.2f}")
    print()
    print("per-guess role breakdown (of every word actually guessed, pooled across both teams):")
    print(
        f"  own {100 * result.guess_own_rate:5.1f}%   opponent {100 * result.guess_opponent_rate:5.1f}%   "
        f"neutral {100 * result.guess_neutral_rate:5.1f}%   assassin {100 * result.guess_assassin_rate:5.1f}%"
    )


if __name__ == "__main__":
    main()
