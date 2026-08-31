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

The guesser call for each active game (`_play_turn`, inside
`_play_batch_group`) runs on a thread pool, not sequentially: clue
selection above is CPU/numpy work that benefits from GPU batching, but
the guesser step doesn't -- and when the guesser is `LLMGuesser`, it's a
blocking network call, so running `batch_size` games' calls one at a
time would leave every one of them idle waiting on the one in front of
it. Threads (not another process pool) are enough here since the work
is I/O-bound, not CPU-bound -- see codenames/guessers/llm.py and
codenames/llm_store.py for the locking that makes one LLMGuesser
instance safe to call from many threads at once without serializing the
network calls themselves.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_search import top_legal_clue
from codenames.codemasters.learned import LearnedCodemaster
from codenames.game import DEFAULT_MAX_TURNS, TurnResult, TwoTeamGameResult, TwoTeamTurnResult
from codenames.game import play_turn as _play_turn
from codenames.gpu_features import build_features_batch_multi
from codenames.guessers import load_pool
from codenames.guessers.base import Guesser
from codenames.guessers.registry import training_pool
from codenames.llm_store import GameRecordStore, board_by_role
from codenames.scorer import expected_reward_and_best_n
from codenames.similarity import SimilarityTensor
from codenames.two_team_arena import MIXED_GUESSER, TwoTeamSelfPlayResult, _new_stats, finalize_result, update_stats


def _winner_if_any(board: Board) -> str | None:
    if board.remaining(Role.OWN) == 0:
        return "A"
    if board.remaining(Role.OPPONENT) == 0:
        return "B"
    return None


def _play_batch_group(
    codemaster: LearnedCodemaster,
    guessers: dict[int, Guesser],
    boards: list[Board],
    sims: SimilarityTensor,
    max_turns: int,
    device: torch.device,
    record_store: GameRecordStore | None = None,
    run_label: str = "",
) -> list[TwoTeamGameResult]:
    """Play every board in `boards` as a two-team game to completion,
    batching the codemaster's clue selection across every game still in
    progress each half-turn. Mirrors codenames.game.play_two_team_game's
    exact win/loss/timeout semantics per game -- see this module's
    docstring for why batching across games is valid here. `guessers` is
    keyed by board seed -- each game can have its own guesser (see
    MIXED_GUESSER in codenames/two_team_arena.py), since the guesser only
    matters after the batched codemaster forward pass, not during it."""
    # Snapshotted before any word is revealed -- boards get mutated in
    # place as the while loop below plays them out.
    by_role_by_seed = {b.seed: board_by_role(b) for b in boards} if record_store is not None else None
    game_results = {b.seed: TwoTeamGameResult(seed=b.seed) for b in boards}
    # Per-game, per-side backlog history (see codenames/guessers/base.py)
    # -- mirrors play_two_team_game's `sides` dict, keyed by seed since
    # many games are driven in lockstep here instead of one at a time.
    history_by_seed: dict[int, dict[str, list[tuple[str, int]]]] = {b.seed: {"A": [], "B": []} for b in boards}
    active: dict[int, Board] = {b.seed: b for b in boards}
    turn_order = ["A", "B"]
    round_index = 0

    # Sized to the whole batch (its largest possible round) and reused
    # across every round in this batch group, rather than spun up fresh
    # each time -- there can be dozens of rounds per game, and thread
    # creation isn't free even though it's cheap next to a network
    # round-trip.
    guesser_pool = ThreadPoolExecutor(max_workers=len(boards))
    try:
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

            # Clue selection is pure CPU/numpy work (cheap) -- computed
            # up front, sequentially, so phase 2 below has nothing left to
            # do per game except the guesser call.
            clue_and_number: dict[int, tuple[str, int]] = {}
            view_by_seed: dict[int, Board | OpponentBoardView] = {}
            for i, seed in enumerate(current_seeds):
                view = views[i]
                view_by_seed[seed] = view
                best_n, scores = expected_reward_and_best_n(
                    probs[i],
                    own_reward=codemaster.own_reward,
                    neutral_reward=codemaster.neutral_reward,
                    opponent_reward=codemaster.opponent_reward,
                    assassin_reward=codemaster.miss_penalty,
                )
                clue = top_legal_clue(sims, view, scores)
                number = int(best_n[sims.clue_index[clue.lower()]])
                clue_and_number[seed] = (clue, number)

            # The guesser call is where a real LLMGuesser blocks on a
            # network round-trip -- run every active game's turn on the
            # thread pool so those round-trips overlap instead of
            # happening one game at a time (see this module's docstring
            # and codenames/guessers/llm.py for the thread-safety this
            # relies on). Each game's Board/history is independent, so
            # there's no shared mutable state across threads here besides
            # the guesser instance(s) themselves.
            def _play_one(seed: int) -> tuple[TurnResult, list[str]]:
                view = view_by_seed[seed]
                candidates_before_turn = [w for w in view.words if not view.is_revealed(w)]
                turn = _play_turn(
                    view, codemaster, guessers[seed], sims, clue_and_number=clue_and_number[seed], history=history_by_seed[seed][team]
                )
                return turn, candidates_before_turn

            turn_by_seed = dict(zip(current_seeds, guesser_pool.map(_play_one, current_seeds)))

            for seed in current_seeds:
                turn, candidates_before_turn = turn_by_seed[seed]
                clue, number = clue_and_number[seed]
                guesser = guessers[seed]
                side_history = history_by_seed[seed][team]
                board = active[seed]
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
    finally:
        guesser_pool.shutdown()

    if record_store is not None:
        for seed, result in game_results.items():
            record_store.add_game(by_role_by_seed[seed], result, label=run_label)

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
    game_record_db: Path | None = None,
    run_label: str = "",
) -> TwoTeamSelfPlayResult:
    """Runs `len(seeds)` two-team games, `codemaster` + the named guesser
    (or `MIXED_GUESSER` -- see codenames/two_team_arena.py) on both sides
    of each, batching clue selection across up to `batch_size`
    simultaneous games.

    `game_record_db`/`run_label`: see run_two_team_self_play's matching
    params in codenames/two_team_arena.py -- same persisted format,
    written from this single process instead of across worker processes."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codemaster.model.to(device)
    codemaster.device = device

    if guesser_name == MIXED_GUESSER:
        pool = list(training_pool(guesser_pool_config).values())
        guesser_for_seed = lambda seed: random.Random(seed).choice(pool)  # noqa: E731
    else:
        single_guesser = load_pool(guesser_pool_config)[guesser_name].guesser
        guesser_for_seed = lambda seed: single_guesser  # noqa: E731
    stats = _new_stats()
    record_store = GameRecordStore(game_record_db) if game_record_db is not None else None

    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        boards = [Board.generate(seed=s) for s in batch_seeds]
        guessers = {s: guesser_for_seed(s) for s in batch_seeds}
        results = _play_batch_group(codemaster, guessers, boards, sims, max_turns, device, record_store, run_label)
        for result in results:
            update_stats(stats, result)

    return finalize_result(stats)
