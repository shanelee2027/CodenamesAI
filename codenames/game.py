"""Game loop.

`play_two_team_game` is a real two-team game: two teams alternate turns on
one board, each spymaster and guesser playing its own side through
`play_turn`. Neither is aware two teams exist -- team B sees the board
through `OpponentBoardView` (board.py), where its words are the "own"
ones. A turn ends the moment a non-own word is revealed or the announced
number of guesses runs out.

Reward, the yardstick every result is measured in: +1 per own word, -0.2
and stop on neutral, -1 and stop on opponent, -10 and stop on assassin.
Neutral is non-zero because it still costs a turn and gains nothing.

A clue announcing `n` gets exactly `n` guesses -- no standard-Codenames
"+1 bonus guess". A guesser has no notion of "still feels confident" to
decide when to spend one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codenames.board import Board, OpponentBoardView, Role
from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor

if TYPE_CHECKING:
    # Deferred: spymasters import this module (for ROLE_REWARD), so
    # importing Spymaster (or TurnContext) at module level here would
    # close that into a circular import. `from __future__ import annotations` (above) already makes
    # every annotation in this file a lazy string, so Spymaster itself is
    # only ever needed by type checkers. `TurnContext` is also
    # constructed at runtime (not just referenced in an annotation), so
    # `play_turn` imports it locally instead -- by the time play_turn is
    # actually called, whatever constructed the `spymaster` argument has
    # already fully imported codenames.spymasters, so the cycle above
    # can't actually happen at that point.
    from codenames.spymasters.base import Spymaster

ROLE_REWARD: dict[Role, float] = {
    Role.OWN: 1.0,
    Role.NEUTRAL: -0.2,
    Role.OPPONENT: -1.0,
    Role.ASSASSIN: -10.0,
}


def role_costs(
    neutral: float | None = None,
    opponent: float | None = None,
    assassin: float | None = None,
) -> dict[Role, float]:
    """What a spymaster *believes* a wrong guess costs, in own-words.

    Deliberately separate from `ROLE_REWARD` above, which is the yardstick:
    the number `play_turn` accumulates and every recorded result is measured
    in. Before this, spymasters read `ROLE_REWARD` directly, which made the
    two the same constant -- so sweeping a spymaster's risk appetite would
    have moved the measuring stick along with the thing being measured, and
    no two runs would have compared. A spymaster's costs are its opinion; the
    game's rewards are the scoreboard.

    `None` means "whatever the scoreboard says", so the default is exactly the
    old behaviour and no existing result changes. Positive magnitudes, since
    every caller took `abs()` of the reward anyway.
    """
    given = {Role.NEUTRAL: neutral, Role.OPPONENT: opponent, Role.ASSASSIN: assassin}
    return {
        role: abs(ROLE_REWARD[role]) if value is None else float(value)
        for role, value in given.items()
    }

# Real games don't have a fixed turn cap, but a guesser that returns no
# ranking could in principle never finish a board. This bounds the worst
# case: BOARD_SIZE turns always suffices if any progress is made at all,
# plus slack for zero-progress turns.
DEFAULT_MAX_TURNS = 40


@dataclass
class TurnResult:
    clue: str
    number: int
    guesses: list[tuple[str, Role]] = field(default_factory=list)
    reward: float = 0.0
    ended_reason: str = ""  # "own_words_complete" | "opponent" | "neutral" | "assassin" | "exhausted_guesses" | "no_guesses"


def play_turn(
    board: Board,
    spymaster: Spymaster,
    guesser: Guesser,
    sims: SimilarityTensor,
) -> TurnResult:
    # Local import: see the module-level TYPE_CHECKING comment above for
    # why TurnContext can't be imported at module scope here.
    from codenames.spymasters.base import TurnContext

    ctx = TurnContext(board=board, turn_index=len(board.revealed))
    clue, number = spymaster.give_clue(ctx, sims)
    candidates = [w for w in board.words if not board.is_revealed(w)]
    attempts = guesser.rank_candidates(clue, candidates, sims, number=number)[:number]

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
    team_a: tuple[Spymaster, Guesser],
    team_b: tuple[Spymaster, Guesser],
    sims: SimilarityTensor,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> TwoTeamGameResult:
    """Two teams alternate turns on one shared board (see board.py's
    module docstring for why no other code needs to change for this):
    team A sees `board` directly (their 9 words are Role.OWN, per
    however its Cards were generated); team B sees the exact same
    physical board through `OpponentBoardView` (their 8 words are
    Role.OWN from that view instead). Neither spymaster nor guesser is
    aware two teams exist -- both just play `play_turn` against whichever
    view they're handed.

    Per real Codenames rules: the 9-card team (A) always moves first
    (verified true of Board.generate -- see docs/log.md's game-setup-
    invariant entry); the game ends the instant either team's own words
    are all revealed -- a win for that team, even if the *other* team's
    guess was what revealed the last one, exactly like an opposing team's
    accidental reveal helps you in the real game -- or either team's
    guess hits the assassin (immediate loss for whoever revealed it, a
    win for the other team). `max_turns` caps each team's own turn count
    (so up to `2 * max_turns` total half-turns) before a timeout, a
    safety valve for a guesser that never guesses."""
    view_b = OpponentBoardView(board)
    sides: dict[str, dict] = {
        "A": {"view": board, "spymaster": team_a[0], "guesser": team_a[1]},
        "B": {"view": view_b, "spymaster": team_b[0], "guesser": team_b[1]},
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

        turn = play_turn(view, side["spymaster"], side["guesser"], sims)
        result.turns.append(TwoTeamTurnResult(team=team, turn=turn))
        result.total_reward[team] += turn.reward

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
