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

    def rank_candidates(self, clue, candidate_words, sims):
        candidates = set(candidate_words)
        return [w for w in self.preferred_order if w in candidates]


class TestPlayTurn:
    def test_all_own_reveals_exhaust_guesses(self):
        board = make_board()
        cm = FixedCodemaster(number=2)
        # Two own words, both guessed -- attempts = number+1 = 3, but only
        # two own words offered so it never hits a miss.
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
