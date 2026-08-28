"""Single-team game loop (SCOPE.md §M6, §8 directory layout).

This project only ever builds the codemaster for one team (see board.py's
module docstring), so there is no opposing team taking its own turns.
"Opponent" and "neutral" words are pure distractors sitting on the same
board; the game is won by revealing every own word before the assassin, and
a turn ends the moment a non-own word is revealed or the codemaster's
attempts run out.

Reward per SCOPE §2 (play-time scoring): +1 per own word, 0 and stop on
neutral, -1 and stop on opponent, -10 and stop on assassin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codenames.board import Board, Role
from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor

if TYPE_CHECKING:
    # Deferred: codemasters/learned.py depends on scorer.py, which depends
    # on this module (for ROLE_REWARD) -- importing Codemaster at runtime
    # here would close that into a circular import. `from __future__ import
    # annotations` (above) already makes every annotation in this file a
    # lazy string, so this is only ever needed by type checkers.
    from codenames.codemasters.base import Codemaster

ROLE_REWARD: dict[Role, float] = {
    Role.OWN: 1.0,
    Role.NEUTRAL: 0.0,
    Role.OPPONENT: -1.0,
    Role.ASSASSIN: -10.0,
}

# Real games don't have a fixed turn cap, but a guesser pool member (e.g. a
# ConfidenceThresholdGuesser that declines every clue) could in principle
# never finish a board. This bounds worst case to at most one word revealed
# per turn -- BOARD_SIZE turns always suffices if any progress is made at
# all -- plus slack for zero-progress turns that still end sensibly.
DEFAULT_MAX_TURNS = 40


@dataclass
class TurnResult:
    clue: str
    number: int
    guesses: list[tuple[str, Role]] = field(default_factory=list)
    reward: float = 0.0
    ended_reason: str = ""  # "own_words_complete" | "opponent" | "neutral" | "assassin" | "exhausted_guesses" | "no_guesses"


@dataclass
class GameResult:
    seed: int
    turns: list[TurnResult] = field(default_factory=list)
    outcome: str = ""  # "win" | "loss" | "timeout"
    total_reward: float = 0.0


def play_turn(board: Board, codemaster: Codemaster, guesser: Guesser, sims: SimilarityTensor) -> TurnResult:
    clue, number = codemaster.give_clue(board, sims)
    candidates = [w for w in board.words if not board.is_revealed(w)]
    ranked = guesser.rank_candidates(clue, candidates, sims)
    attempts = ranked[: number + 1]

    if not attempts:
        return TurnResult(clue=clue, number=number, ended_reason="no_guesses")

    result = TurnResult(clue=clue, number=number)
    for word in attempts:
        role = board.reveal(word)
        result.guesses.append((word, role))
        result.reward += ROLE_REWARD[role]

        if role == Role.ASSASSIN:
            result.ended_reason = "assassin"
            return result
        if role != Role.OWN:
            result.ended_reason = role.value
            return result
        if board.remaining(Role.OWN) == 0:
            result.ended_reason = "own_words_complete"
            return result

    result.ended_reason = "exhausted_guesses"
    return result


def play_game(
    board: Board,
    codemaster: Codemaster,
    guesser: Guesser,
    sims: SimilarityTensor,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> GameResult:
    result = GameResult(seed=board.seed)

    for _ in range(max_turns):
        if board.remaining(Role.OWN) == 0:
            result.outcome = "win"
            break

        turn = play_turn(board, codemaster, guesser, sims)
        result.turns.append(turn)
        result.total_reward += turn.reward

        if turn.ended_reason == "assassin":
            result.outcome = "loss"
            break
        if board.remaining(Role.OWN) == 0:
            result.outcome = "win"
            break
    else:
        result.outcome = "timeout"

    return result
