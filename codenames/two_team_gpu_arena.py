"""GPU-batched two-team self-play arena for LearnedCodemaster specifically
-- plays many simultaneous two-team games in lockstep on one GPU process,
mirroring codenames/gpu_arena.py's single-team batching (see that
module's docstring for the underlying "batch the forward pass across
many boards" idea and its measured speedup) but doubled for two sides.

Why batching across *games* still works here even though a two-team game
alternates turns internally: every active game advances by exactly one
half-turn per outer-loop iteration below, and games only ever leave the
active set by finishing (never by getting out of sync with the others).
So at any iteration, every still-active game is on the *same* team's
turn -- that shared parity is what makes it possible to gather the
correct board perspective (the real Board for team A, its
OpponentBoardView for team B -- see codenames/board.py) across every
active game and run one batched forward pass, exactly like the
single-team path already does for its one perspective.

Only accelerates the case codenames/two_team_arena.py's own docstring
already scopes to: bulk two-team *self-play*, the SAME LearnedCodemaster
+ guesser pair on both sides. A mixed-codemaster or baseline-only
two-team comparison stays on scripts/run_two_team_arena.py's
process-parallel path -- nothing here to batch for a codemaster that
doesn't score the whole clue vocabulary per turn.

Reuses codenames/two_team_arena.py's stats bookkeeping
(_new_stats/update_stats/finalize_result) unchanged -- only clue
*selection* is batched here, nothing about how a turn plays out or how
results are summarized once a clue is chosen.
"""

from __future__ import annotations

from pathlib import Path

import torch

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_search import top_legal_clue
from codenames.codemasters.learned import LearnedCodemaster
from codenames.game import DEFAULT_MAX_TURNS, TwoTeamGameResult, TwoTeamTurnResult
from codenames.game import play_turn as _play_turn
from codenames.gpu_features import build_features_batch_multi
from codenames.guessers import load_pool
from codenames.guessers.base import Guesser
from codenames.scorer import expected_reward_and_best_n
from codenames.similarity import SimilarityTensor
from codenames.two_team_arena import TwoTeamSelfPlayResult, _new_stats, finalize_result, update_stats


def _winner_if_any(board: Board) -> str | None:
    if board.remaining(Role.OWN) == 0:
        return "A"
    if board.remaining(Role.OPPONENT) == 0:
        return "B"
    return None


def _play_batch_group(
    codemaster: LearnedCodemaster,
    guesser: Guesser,
    boards: list[Board],
    sims: SimilarityTensor,
    max_turns: int,
    device: torch.device,
) -> list[TwoTeamGameResult]:
    """Play every board in `boards` as a two-team game to completion,
    batching the codemaster's clue selection across every game still in
    progress each half-turn. Mirrors codenames.game.play_two_team_game's
    exact win/loss/timeout semantics per game -- see this module's
    docstring for why batching across games is valid here."""
    game_results = {b.seed: TwoTeamGameResult(seed=b.seed) for b in boards}
    # Per-game, per-side backlog history (see codenames/guessers/base.py)
    # -- mirrors play_two_team_game's `sides` dict, keyed by seed since
    # many games are driven in lockstep here instead of one at a time.
    history_by_seed: dict[int, dict[str, list[tuple[str, int]]]] = {b.seed: {"A": [], "B": []} for b in boards}
    active: dict[int, Board] = {b.seed: b for b in boards}
    turn_order = ["A", "B"]
    round_index = 0

    while active:
        if round_index >= max_turns * 2:
            for seed in active:
                game_results[seed].outcome = "timeout"
            break

        # Every active game shares this round's team -- see module
        # docstring for why that invariant holds.
        team = turn_order[round_index % 2]
        current_seeds = list(active.keys())
        views = [active[s] if team == "A" else OpponentBoardView(active[s]) for s in current_seeds]
        turn_indices = [len(v.revealed) for v in views]

        features = build_features_batch_multi(sims, views, turn_indices, device)  # (n, n_clues, dim)
        n_active, n_clues, dim = features.shape
        with torch.no_grad():
            probs = codemaster.model.predict_proba(features.reshape(n_active * n_clues, dim).to(codemaster.device))
        probs = probs.cpu().numpy().reshape(n_active, n_clues, -1)

        for i, seed in enumerate(current_seeds):
            board = active[seed]
            view = views[i]
            best_n, scores = expected_reward_and_best_n(
                probs[i],
                own_reward=codemaster.own_reward,
                neutral_reward=codemaster.neutral_reward,
                opponent_reward=codemaster.opponent_reward,
                assassin_reward=codemaster.miss_penalty,
            )
            clue = top_legal_clue(sims, view, scores)
            number = int(best_n[sims.clue_index[clue.lower()]])

            candidates_before_turn = [w for w in view.words if not view.is_revealed(w)]
            side_history = history_by_seed[seed][team]
            turn = _play_turn(view, codemaster, guesser, sims, clue_and_number=(clue, number), history=side_history)
            result = game_results[seed]
            result.turns.append(TwoTeamTurnResult(team=team, turn=turn))
            result.total_reward[team] += turn.reward
            history_by_seed[seed][team] = guesser.update_history(
                side_history, clue, number, turn, candidates_before_turn, sims
            )

            if turn.ended_reason == "assassin":
                result.outcome = "loss"
                result.winner = "B" if team == "A" else "A"
                del active[seed]
                continue
            winner = _winner_if_any(board)
            if winner is not None:
                result.outcome = "win"
                result.winner = winner
                del active[seed]

        round_index += 1

    return list(game_results.values())


def run_two_team_self_play_gpu(
    codemaster: LearnedCodemaster,
    guesser_pool_config: Path,
    guesser_name: str,
    seeds: list[int],
    sims: SimilarityTensor,
    batch_size: int = 32,
    max_turns: int = DEFAULT_MAX_TURNS,
    device: torch.device | None = None,
) -> TwoTeamSelfPlayResult:
    """Runs `len(seeds)` two-team games, `codemaster` + the named guesser
    on both sides of each, batching clue selection across up to
    `batch_size` simultaneous games."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codemaster.model.to(device)
    codemaster.device = device

    guesser = load_pool(guesser_pool_config)[guesser_name].guesser
    stats = _new_stats()

    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        boards = [Board.generate(seed=s) for s in batch_seeds]
        results = _play_batch_group(codemaster, guesser, boards, sims, max_turns, device)
        for result in results:
            update_stats(stats, result)

    return finalize_result(stats)
