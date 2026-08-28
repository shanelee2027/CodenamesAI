from __future__ import annotations

from codenames.board import Board, Card, Role
from codenames.codemasters.base import Codemaster
from codenames.game import ROLE_REWARD, play_game, play_turn
from codenames.guessers.base import Guesser

BOARD_WORDS = [f"Board{i}" for i in range(25)]


def make_board(revealed: list[str] | None = None) -> Board:
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(BOARD_WORDS, roles))
    board = Board(cards=cards, seed=1)
    for w in revealed or []:
        board.reveal(w)
    return board


class FixedCodemaster(Codemaster):
    def __init__(self, clue: str = "clue", number: int = 1):
        self.clue = clue
        self.number = number

    def give_clue(self, board, sims):
        return self.clue, self.number


class ScriptedGuesser(Guesser):
    """Ranks whatever candidates are given in a fixed preferred order,
    ignoring the clue -- lets tests dictate exactly what gets guessed."""

    def __init__(self, preferred_order: list[str]):
        self.preferred_order = preferred_order

    def score_candidates(self, clue, candidate_words, sims):
        return {w: -self.preferred_order.index(w) if w in self.preferred_order else float("-inf") for w in candidate_words}

    def rank_candidates(self, clue, candidate_words, sims, number=None, history=None):
        candidates = set(candidate_words)
        return [w for w in self.preferred_order if w in candidates]


class TestPlayTurn:
    def test_all_own_reveals_exhaust_guesses(self):
        board = make_board()
        cm = FixedCodemaster(number=2)
        # Two own words, both guessed -- attempts = number = 2, exactly
        # matching the two own words offered so it never hits a miss.
        guesser = ScriptedGuesser(["Board0", "Board1"])
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "exhausted_guesses"
        assert [w for w, _ in turn.guesses] == ["Board0", "Board1"]
        assert turn.reward == 2.0

    def test_stops_on_first_non_own(self):
        board = make_board()
        cm = FixedCodemaster(number=3)
        guesser = ScriptedGuesser(["Board0", "Board9", "Board1"])  # Board9 is OPPONENT
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "opponent"
        assert [w for w, _ in turn.guesses] == ["Board0", "Board9"]
        assert turn.reward == ROLE_REWARD[Role.OWN] + ROLE_REWARD[Role.OPPONENT]

    def test_stops_immediately_on_assassin(self):
        board = make_board()
        cm = FixedCodemaster(number=3)
        guesser = ScriptedGuesser(["Board24", "Board0"])  # Board24 is ASSASSIN
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "assassin"
        assert turn.guesses == [("Board24", Role.ASSASSIN)]
        assert turn.reward == ROLE_REWARD[Role.ASSASSIN]

    def test_stops_when_own_words_complete_mid_attempt(self):
        # Only one own word left unrevealed; guessing it should end the
        # turn immediately even though more attempts remained.
        board = make_board(revealed=BOARD_WORDS[:8])
        cm = FixedCodemaster(number=3)
        guesser = ScriptedGuesser(["Board8", "Board9"])
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "own_words_complete"
        assert [w for w, _ in turn.guesses] == ["Board8"]

    def test_no_guesses_when_guesser_declines(self):
        board = make_board()
        cm = FixedCodemaster(number=1)
        guesser = ScriptedGuesser([])
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "no_guesses"
        assert turn.guesses == []
        assert turn.reward == 0.0


class TestPlayGame:
    def test_wins_when_all_own_words_revealed(self):
        board = make_board()
        cm = FixedCodemaster(number=8)
        guesser = ScriptedGuesser(BOARD_WORDS[:9])  # all own words, in order
        result = play_game(board, cm, guesser, sims=None, max_turns=5)
        assert result.outcome == "win"
        assert board.remaining(Role.OWN) == 0

    def test_loses_on_assassin(self):
        board = make_board()
        cm = FixedCodemaster(number=1)
        guesser = ScriptedGuesser(["Board0", "Board24"])
        result = play_game(board, cm, guesser, sims=None, max_turns=5)
        assert result.outcome == "loss"

    def test_times_out_when_guesser_never_guesses(self):
        board = make_board()
        cm = FixedCodemaster(number=1)
        guesser = ScriptedGuesser([])
        result = play_game(board, cm, guesser, sims=None, max_turns=3)
        assert result.outcome == "timeout"
        assert len(result.turns) == 3


class SequencedCodemaster(Codemaster):
    """Gives a different (clue, number) each call, in order -- lets a
    test simulate a real multi-turn game instead of FixedCodemaster's one
    repeated clue."""

    def __init__(self, plan: list[tuple[str, int]]):
        self.plan = list(plan)
        self.calls = 0

    def give_clue(self, board, sims):
        clue_and_number = self.plan[self.calls]
        self.calls += 1
        return clue_and_number


class AlwaysBonusGuesser(ScriptedGuesser):
    """Claims the bonus guess whenever any backlog exists at all,
    regardless of content -- isolates testing the game loop's history
    threading (this class) from testing HistoryAwareGuesser's own
    z-score-based decision of *whether* to claim it
    (tests/test_guessers.py)."""

    def bonus_guesses(self, clue, candidate_words, sims, number, history=None):
        return 1 if history else 0


class TestBonusGuessThreading:
    """codenames/guessers/base.py's backlog/bonus-guess mechanism,
    exercised through play_turn/play_game rather than through a specific
    guesser's decision logic."""

    def test_a_miss_creates_backlog_and_a_later_bonus_spends_it(self):
        board = make_board()
        # Turn 1: number=2, but the 2nd guess is a miss (Board9 is
        # OPPONENT) -- ends after 1 correct guess, leaving 2-1=1 word
        # believed owed by "c1".
        cm = SequencedCodemaster([("c1", 2), ("c2", 1)])
        guesser = AlwaysBonusGuesser(["Board0", "Board9", "Board1", "Board2"])
        result = play_game(board, cm, guesser, sims=None, max_turns=2)

        turn1 = result.turns[0]
        assert turn1.ended_reason == "opponent"
        assert [w for w, _ in turn1.guesses] == ["Board0", "Board9"]

        # Turn 2 announces number=1, but AlwaysBonusGuesser claims the
        # bonus since turn 1 left backlog -- should attempt 2 guesses,
        # not 1.
        turn2 = result.turns[1]
        assert [w for w, _ in turn2.guesses] == ["Board1", "Board2"]

    def test_no_backlog_means_no_bonus_even_for_a_bonus_claiming_guesser(self):
        # Same guesser class, but the first clue's number is fully used
        # up by correct guesses (no miss) -- no backlog, so turn 2 should
        # still get exactly its announced number.
        board = make_board()
        cm = SequencedCodemaster([("c1", 2), ("c2", 1)])
        guesser = AlwaysBonusGuesser(["Board0", "Board1", "Board2", "Board3"])
        result = play_game(board, cm, guesser, sims=None, max_turns=2)

        assert result.turns[0].ended_reason == "exhausted_guesses"
        turn2 = result.turns[1]
        assert [w for w, _ in turn2.guesses] == ["Board2"]
