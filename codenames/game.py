"""Single-team game loop (SCOPE.md §M6, §8 directory layout).

This project only ever builds the codemaster for one team (see board.py's
module docstring), so there is no opposing team taking its own turns.
"Opponent" and "neutral" words are pure distractors sitting on the same
board; the game is won by revealing every own word before the assassin, and
a turn ends the moment a non-own word is revealed or the codemaster's
attempts run out.

Reward per SCOPE §2 (play-time scoring): +1 per own word, -0.2 and stop
on neutral, -1 and stop on opponent, -10 and stop on assassin. Neutral
being non-zero (rather than a true no-op) is deliberate: it still costs a
turn and reveals no information toward winning, so it should be mildly
penalized rather than treated as free -- see docs/log.md.

A clue announcing `n` gets exactly `n` guesses by default -- no automatic
standard-Codenames "+1 bonus guess" (see docs/log.md's numbering-
convention entries for why that was dropped). A guesser can still claim
one extra guess this turn via `Guesser.bonus_guesses` (see
codenames/guessers/base.py), but only if it has an actual, tracked reason
to -- e.g. `HistoryAwareGuesser` believes a past clue's miss left a word
unaccounted-for. `codenames/scorer.py`'s reward math is unaware of this:
it still assumes exactly `n` attempts, so real play with a bonus-claiming
guesser slightly outperforms what a codemaster's own expected-reward
calculation predicts for it, never the other way around.
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
    Role.NEUTRAL: -0.2,
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


def play_turn(
    board: Board,
    codemaster: Codemaster,
    guesser: Guesser,
    sims: SimilarityTensor,
    clue_and_number: tuple[str, int] | None = None,
    history: list[tuple[str, int]] | None = None,
) -> TurnResult:
    """`clue_and_number`, if given, skips calling `codemaster.give_clue()`
    and uses that pair directly -- lets a caller compute the clue for many
    boards at once (batched, off the hot path of this function) and still
    reuse this exact tested attempt/reveal/stop logic per board. See
    codenames/gpu_arena.py, which batches LearnedCodemaster's give_clue()
    across many simultaneous games on GPU for a real throughput win, then
    drives each board's turn through this same function unchanged.

    `history`, if given, is the backlog state from `Guesser.update_history`
    (see codenames/guessers/base.py) -- forwarded to the guesser so it can
    claim a bonus guess beyond `number` if it has a real reason to
    (default: 0, i.e. every guesser that doesn't override `bonus_guesses`
    plays exactly as it always has)."""
    clue, number = clue_and_number if clue_and_number is not None else codemaster.give_clue(board, sims)
    candidates = [w for w in board.words if not board.is_revealed(w)]
    bonus = guesser.bonus_guesses(clue, candidates, sims, number, history=history)
    ranked = guesser.rank_candidates(clue, candidates, sims, number=number, history=history)
    attempts = ranked[: number + bonus]

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
    history: list[tuple[str, int]] = []

    for _ in range(max_turns):
        if board.remaining(Role.OWN) == 0:
            result.outcome = "win"
            break

        # Snapshotted before the turn runs (play_turn recomputes the same
        # thing internally -- kept separate rather than having play_turn
        # return it too, so its return type stays unchanged for every
        # other caller).
        candidates_before_turn = [w for w in board.words if not board.is_revealed(w)]
        turn = play_turn(board, codemaster, guesser, sims, history=history)
        result.turns.append(turn)
        result.total_reward += turn.reward
        history = guesser.update_history(history, turn.clue, turn.number, turn, candidates_before_turn, sims)

        if turn.ended_reason == "assassin":
            result.outcome = "loss"
            break
        if board.remaining(Role.OWN) == 0:
            result.outcome = "win"
            break
    else:
        result.outcome = "timeout"

    return result
