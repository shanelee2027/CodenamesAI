"""Game loop (SCOPE.md §M6, §8 directory layout).

`play_game` is single-team: "opponent" and "neutral" words just sit on
the board as pure distractors, nobody actively pursuing them -- most of
the codebase (arena evaluations, training data generation) only ever
uses this, since every codemaster/guesser is written against "own" as a
fixed perspective (see board.py's module docstring). `play_two_team_game`
is real two-team play, added later without changing any codemaster,
guesser, or the scorer -- see its own docstring and `OpponentBoardView`
in board.py for how. In both, a turn ends the moment a non-own word is
revealed or the codemaster's attempts run out.

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

from codenames.board import Board, OpponentBoardView, Role
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


@dataclass
class TwoTeamTurnResult:
    team: str  # "A" | "B"
    turn: TurnResult


@dataclass
class TwoTeamGameResult:
    seed: int
    turns: list[TwoTeamTurnResult] = field(default_factory=list)
    outcome: str = ""  # "win" | "loss" | "timeout" -- "loss" specifically means the assassin ended it
    winner: str | None = None  # "A" | "B", or None on a timeout
    total_reward: dict[str, float] = field(default_factory=lambda: {"A": 0.0, "B": 0.0})


def play_two_team_game(
    board: Board,
    team_a: tuple[Codemaster, Guesser],
    team_b: tuple[Codemaster, Guesser],
    sims: SimilarityTensor,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> TwoTeamGameResult:
    """Two teams alternate turns on one shared board (see board.py's
    module docstring for why no other code needs to change for this):
    team A sees `board` directly (their 9 words are Role.OWN, per
    however its Cards were generated); team B sees the exact same
    physical board through `OpponentBoardView` (their 8 words are
    Role.OWN from that view instead). Neither codemaster nor guesser is
    aware two teams exist -- both just play `play_turn` against whichever
    view they're handed, identically to single-team play, each with
    their own independent backlog `history` (see
    codenames/guessers/base.py) so a HistoryAwareGuesser on one side
    can't see or be confused by the other side's misses.

    Per real Codenames rules: the 9-card team (A) always moves first
    (verified true of Board.generate -- see docs/log.md's game-setup-
    invariant entry); the game ends the instant either team's own words
    are all revealed -- a win for that team, even if the *other* team's
    guess was what revealed the last one, exactly like an opposing team's
    accidental reveal helps you in the real game -- or either team's
    guess hits the assassin (immediate loss for whoever revealed it, a
    win for the other team). `max_turns` caps each team's own turn count
    (so up to `2 * max_turns` total half-turns) before a timeout,
    mirroring play_game's guesser-that-never-guesses safety valve."""
    view_b = OpponentBoardView(board)
    sides: dict[str, dict] = {
        "A": {"view": board, "codemaster": team_a[0], "guesser": team_a[1], "history": []},
        "B": {"view": view_b, "codemaster": team_b[0], "guesser": team_b[1], "history": []},
    }

    def _winner_if_any() -> str | None:
        if sides["A"]["view"].remaining(Role.OWN) == 0:
            return "A"
        if sides["B"]["view"].remaining(Role.OWN) == 0:
            return "B"
        return None

    result = TwoTeamGameResult(seed=board.seed)
    turn_order = ["A", "B"]
    for half_turn in range(max_turns * 2):
        team = turn_order[half_turn % 2]
        side = sides[team]
        view = side["view"]

        candidates_before_turn = [w for w in view.words if not view.is_revealed(w)]
        turn = play_turn(view, side["codemaster"], side["guesser"], sims, history=side["history"])
        result.turns.append(TwoTeamTurnResult(team=team, turn=turn))
        result.total_reward[team] += turn.reward
        side["history"] = side["guesser"].update_history(
            side["history"], turn.clue, turn.number, turn, candidates_before_turn, sims
        )

        if turn.ended_reason == "assassin":
            result.outcome = "loss"
            result.winner = "B" if team == "A" else "A"
            return result

        winner = _winner_if_any()
        if winner is not None:
            result.outcome = "win"
            result.winner = winner
            return result

    result.outcome = "timeout"
    return result
