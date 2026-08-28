"""GPU-batched arena runner for LearnedCodemaster specifically -- plays
many simultaneous games in lockstep on one GPU process, instead of
codenames/arena.py::run_arena's one-board-per-OS-process model.

Why this exists (measured, see docs/log.md's GPU-arena entries): the
dominant per-turn cost for LearnedCodemaster is scoring the entire clue
vocabulary (~111k clues) -- both the feature construction
(codenames/gpu_features.py) and the model's forward pass batch far better
across many boards at once than across separate OS processes each doing
one board at a time. Measured ~24x per-turn compute speedup at a 32-board
batch (still improving, not yet flattened), translating to roughly 3x
higher realistic arena throughput once you account for run_arena already
getting real parallelism from 8 CPU worker processes.

Only accelerates LearnedCodemaster. Every other codemaster (random,
centroid, oracle, linear_scorer) is already cheap -- it scores a handful
of candidates via a much smaller computation, not the full vocabulary --
so there's nothing here for them to gain, and codenames/arena.py::run_arena
remains the right tool for those. A single invocation can mix both: run
baselines through run_arena, the learned codemaster through this module,
and merge the resulting CrossPlayResult dicts for one combined report
(see scripts/run_arena.py's --gpu-batch-size).

Reuses codenames/game.py::play_turn unchanged (via its clue_and_number
param) for the actual attempt/reveal/stop logic per board, and
codenames/arena.py's new_stats_accumulator/update_stats/finalize_result
for identical stats bookkeeping to run_arena -- only clue *selection* is
batched here, nothing about how a turn plays out once a clue is chosen.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import torch

from codenames.arena import CrossPlayResult, _init_db, _log_game, finalize_result, new_stats_accumulator, update_stats
from codenames.board import Board, Role
from codenames.clue_search import top_legal_clue
from codenames.codemasters.learned import LearnedCodemaster
from codenames.game import DEFAULT_MAX_TURNS, GameResult
from codenames.game import play_turn as _play_turn
from codenames.gpu_features import build_features_batch_multi
from codenames.guessers import load_pool
from codenames.guessers.base import Guesser
from codenames.scorer import expected_reward_and_best_n
from codenames.similarity import SimilarityTensor


def _play_batch_group(
    codemaster: LearnedCodemaster,
    guesser: Guesser,
    boards: list[Board],
    sims: SimilarityTensor,
    max_turns: int,
    device: torch.device,
) -> list[GameResult]:
    """Play every board in `boards` to completion, batching the
    codemaster's clue selection across all boards still in progress each
    round. Mirrors codenames.game.play_game's exact win/loss/timeout
    semantics per board -- see this module's docstring."""
    game_results = {b.seed: GameResult(seed=b.seed) for b in boards}
    turn_count = {b.seed: 0 for b in boards}
    active: dict[int, Board] = {b.seed: b for b in boards}

    while active:
        for seed in [s for s, b in active.items() if turn_count[s] >= max_turns]:
            game_results[seed].outcome = "timeout"
            del active[seed]
        if not active:
            break

        for seed in [s for s, b in active.items() if b.remaining(Role.OWN) == 0]:
            game_results[seed].outcome = "win"
            del active[seed]
        if not active:
            break

        current = list(active.values())
        turn_indices = [len(b.revealed) for b in current]
        features = build_features_batch_multi(sims, current, turn_indices, device)  # (n, n_clues, dim)
        n_active, n_clues, dim = features.shape
        with torch.no_grad():
            probs = codemaster.model.predict_proba(features.reshape(n_active * n_clues, dim).to(codemaster.device))
        probs = probs.cpu().numpy().reshape(n_active, n_clues, -1)

        for i, board in enumerate(current):
            best_n, scores = expected_reward_and_best_n(
                probs[i],
                own_reward=codemaster.own_reward,
                neutral_reward=codemaster.neutral_reward,
                opponent_reward=codemaster.opponent_reward,
                assassin_reward=codemaster.miss_penalty,
            )
            clue = top_legal_clue(sims, board, scores)
            number = int(best_n[sims.clue_index[clue.lower()]])

            turn = _play_turn(board, codemaster, guesser, sims, clue_and_number=(clue, number))
            result = game_results[board.seed]
            result.turns.append(turn)
            result.total_reward += turn.reward
            turn_count[board.seed] += 1

            if turn.ended_reason == "assassin":
                result.outcome = "loss"
                del active[board.seed]
            elif board.remaining(Role.OWN) == 0:
                result.outcome = "win"
                del active[board.seed]

    return list(game_results.values())


def run_arena_gpu(
    codemaster: LearnedCodemaster,
    codemaster_name: str,
    guesser_pool_config: Path,
    seeds: list[int],
    db_path: Path,
    sims: SimilarityTensor,
    batch_size: int = 32,
    max_turns: int = DEFAULT_MAX_TURNS,
    device: torch.device | None = None,
) -> dict[str, CrossPlayResult]:
    """Play `codemaster` against every guesser in the pool, over `seeds`,
    batching clue selection across up to `batch_size` simultaneous games.
    Returns results keyed by guesser name -- combine with run_arena's
    (codemaster, guesser)-keyed dict by prefixing with codemaster_name to
    merge into one report."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codemaster.model.to(device)
    codemaster.device = device

    pool = load_pool(guesser_pool_config)
    conn = _init_db(db_path)
    stats: dict[str, dict[str, float]] = defaultdict(new_stats_accumulator)

    for g_name, entry in pool.items():
        for start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[start : start + batch_size]
            boards = [Board.generate(seed=s) for s in batch_seeds]
            results = _play_batch_group(codemaster, entry.guesser, boards, sims, max_turns, device)
            for result in results:
                _log_game(conn, codemaster_name, g_name, entry.held_out, result)
                update_stats(stats[g_name], result)

    conn.commit()
    conn.close()

    return {g_name: finalize_result(codemaster_name, g_name, pool[g_name].held_out, s) for g_name, s in stats.items()}
