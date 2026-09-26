from __future__ import annotations

from codenames.board import Board, Card, Role
from codenames.spymasters.base import Spymaster
from codenames.game import ROLE_REWARD, STOP, play_turn, play_two_team_game
from codenames.guessers.base import Guesser

BOARD_WORDS = [f"Board{i}" for i in range(25)]


def make_board(revealed: list[str] | None = None) -> Board:
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(BOARD_WORDS, roles))
    board = Board(cards=cards, seed=1)
    for w in revealed or []:
        board.reveal(w)
    return board


class FixedSpymaster(Spymaster):
    def __init__(self, clue: str = "clue", number: int = 1):
        self.clue = clue
        self.number = number

    def top_clues(self, ctx, sims, k):
        return [(self.clue, self.number, 0.0)]


class ScriptedGuesser(Guesser):
    """Ranks whatever candidates are given in a fixed preferred order,
    ignoring the clue -- lets tests dictate exactly what gets guessed."""

    def __init__(self, preferred_order: list[str]):
        self.preferred_order = preferred_order

    def score_candidates(self, clue, candidate_words, sims):
        return {w: -self.preferred_order.index(w) if w in self.preferred_order else float("-inf") for w in candidate_words}

    def rank_candidates(self, clue, candidate_words, sims, number=None):
        candidates = set(candidate_words)
        return [w for w in self.preferred_order if w in candidates]


class TestPlayTurn:
    def test_all_own_reveals_exhaust_guesses(self):
        board = make_board()
        cm = FixedSpymaster(number=2)
        # Two own words, both guessed -- attempts = number = 2, exactly
        # matching the two own words offered so it never hits a miss.
        guesser = ScriptedGuesser(["Board0", "Board1"])
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "exhausted_guesses"
        assert [w for w, _ in turn.guesses] == ["Board0", "Board1"]
        assert turn.reward == 2.0

    def test_stops_on_first_non_own(self):
        board = make_board()
        cm = FixedSpymaster(number=3)
        guesser = ScriptedGuesser(["Board0", "Board9", "Board1"])  # Board9 is OPPONENT
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "opponent"
        assert [w for w, _ in turn.guesses] == ["Board0", "Board9"]
        assert turn.reward == ROLE_REWARD[Role.OWN] + ROLE_REWARD[Role.OPPONENT]

    def test_stops_immediately_on_assassin(self):
        board = make_board()
        cm = FixedSpymaster(number=3)
        guesser = ScriptedGuesser(["Board24", "Board0"])  # Board24 is ASSASSIN
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "assassin"
        assert turn.guesses == [("Board24", Role.ASSASSIN)]
        assert turn.reward == ROLE_REWARD[Role.ASSASSIN]

    def test_stops_when_own_words_complete_mid_attempt(self):
        # Only one own word left unrevealed; guessing it should end the
        # turn immediately even though more attempts remained.
        board = make_board(revealed=BOARD_WORDS[:8])
        cm = FixedSpymaster(number=3)
        guesser = ScriptedGuesser(["Board8", "Board9"])
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "own_words_complete"
        assert [w for w, _ in turn.guesses] == ["Board8"]

    def test_no_guesses_when_guesser_declines(self):
        board = make_board()
        cm = FixedSpymaster(number=1)
        guesser = ScriptedGuesser([])
        turn = play_turn(board, cm, guesser, sims=None)
        assert turn.ended_reason == "no_guesses"
        assert turn.guesses == []
        assert turn.reward == 0.0


class RawGuesser(Guesser):
    """Returns a fixed list verbatim, STOP included -- what a stop-capable
    guesser hands the game."""

    def __init__(self, ranking: list[str]):
        self.ranking = ranking

    def score_candidates(self, clue, candidate_words, sims):
        return {}

    def rank_candidates(self, clue, candidate_words, sims, number=None):
        return list(self.ranking)


class TestStop:
    def test_stop_ends_the_turn_after_the_words_above_it(self):
        turn = play_turn(make_board(), FixedSpymaster(number=3), RawGuesser(["Board0", STOP, "Board1"]), sims=None)
        assert turn.ended_reason == "stopped"
        assert [w for w, _ in turn.guesses] == ["Board0"]
        assert turn.reward == ROLE_REWARD[Role.OWN]

    def test_stop_first_is_a_pass_worth_nothing(self):
        board = make_board()
        turn = play_turn(board, FixedSpymaster(number=2), RawGuesser([STOP, "Board0"]), sims=None)
        assert turn.ended_reason == "stopped" and turn.guesses == [] and turn.reward == 0.0
        assert not board.revealed

    def test_stop_below_the_number_changes_nothing(self):
        turn = play_turn(make_board(), FixedSpymaster(number=2), RawGuesser(["Board0", "Board1", STOP]), sims=None)
        assert turn.ended_reason == "exhausted_guesses"
        assert [w for w, _ in turn.guesses] == ["Board0", "Board1"]

    def test_a_miss_above_stop_still_ends_the_turn_as_a_miss(self):
        turn = play_turn(make_board(), FixedSpymaster(number=3), RawGuesser(["Board9", STOP]), sims=None)
        assert turn.ended_reason == "opponent"


class TestPlayTwoTeamGame:
    """Board0-8 = team A's own (9), Board9-16 = team B's own (8),
    Board17-23 = neutral, Board24 = assassin (see make_board)."""

    def test_team_a_moves_first(self):
        board = make_board()
        team_a = (FixedSpymaster("c", 1), ScriptedGuesser([]))
        team_b = (FixedSpymaster("c", 1), ScriptedGuesser([]))
        result = play_two_team_game(board, team_a, team_b, sims=None, max_turns=1)
        assert result.turns[0].team == "A"
        assert result.turns[1].team == "B"

    def test_team_wins_by_clearing_their_own_words_on_their_own_turn(self):
        board = make_board()
        team_a = (FixedSpymaster("c", 9), ScriptedGuesser(BOARD_WORDS[:9]))  # all 9 of A's own words
        team_b = (FixedSpymaster("c", 1), ScriptedGuesser([]))
        result = play_two_team_game(board, team_a, team_b, sims=None, max_turns=5)
        assert result.outcome == "win"
        assert result.winner == "A"
        assert len(result.turns) == 1  # game ends right after A's first turn, B never moves

    def test_the_other_teams_mistake_can_win_the_game(self):
        # Team A repeatedly (mis)guesses team B's own words (opponent
        # from A's view) one at a time -- each ends A's turn immediately,
        # but after enough of A's turns, B's own words run out and B
        # wins without ever making a guess themselves, exactly like an
        # opposing team's accidental reveal helps you in the real game.
        board = make_board()
        b_own_words = BOARD_WORDS[9:17]  # Board9..Board16, team B's 8 own words
        team_a = (FixedSpymaster("c", 1), ScriptedGuesser(b_own_words))
        team_b = (FixedSpymaster("c", 1), ScriptedGuesser([]))  # never guesses
        result = play_two_team_game(board, team_a, team_b, sims=None, max_turns=20)
        assert result.outcome == "win"
        assert result.winner == "B"
        assert sum(1 for t in result.turns if t.team == "A") == 8
        assert all(t.turn.ended_reason == "opponent" for t in result.turns if t.team == "A")

    def test_assassin_ends_the_game_immediately_and_the_other_team_wins(self):
        board = make_board()
        team_a = (FixedSpymaster("c", 1), ScriptedGuesser(["Board24"]))  # the assassin
        team_b = (FixedSpymaster("c", 1), ScriptedGuesser([]))
        result = play_two_team_game(board, team_a, team_b, sims=None, max_turns=5)
        assert result.outcome == "loss"
        assert result.winner == "B"
        assert len(result.turns) == 1

    def test_times_out_when_neither_team_ever_guesses(self):
        board = make_board()
        team_a = (FixedSpymaster("c", 1), ScriptedGuesser([]))
        team_b = (FixedSpymaster("c", 1), ScriptedGuesser([]))
        result = play_two_team_game(board, team_a, team_b, sims=None, max_turns=3)
        assert result.outcome == "timeout"
        assert result.winner is None
        assert len(result.turns) == 6  # 3 turns each

