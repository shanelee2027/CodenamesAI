"""The LLM spymaster (codenames/spymasters/llm_spymaster.py), against a fake
client: what it is shown, what it accepts, and that it never invents a clue."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codenames.board import Board, OpponentBoardView, Role
from codenames.spymasters.base import TurnContext
from codenames.spymasters.llm_spymaster import LLMSpymaster, board_prompt, parse_clue


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.messages = self

    def create(self, **request):
        self.prompts.append(request["messages"][0]["content"])
        text = self.answers.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=20))


def _board():
    return Board.generate(seed=3)


def _spy(answers, tmp_path=None):
    return LLMSpymaster(client=FakeClient(answers),
                        cache_path=None if tmp_path is None else tmp_path / "s.db")


class TestPrompt:
    def test_lists_each_role_from_the_spymasters_side(self):
        b = _board()
        p = board_prompt(b)
        for w in b.words_by_role(Role.OWN):
            assert w in p.split("Your team's words")[1].split("\n")[0]
        flipped = board_prompt(OpponentBoardView(b))
        for w in b.words_by_role(Role.OPPONENT):
            assert w in flipped.split("Your team's words")[1].split("\n")[0]

    def test_revealed_words_leave_the_role_lists(self):
        b = _board()
        w = b.words_by_role(Role.OWN)[0]
        b.reveal(w)
        p = board_prompt(b)
        assert w not in p.split("Your team's words")[1].split("\n")[0]
        assert w in p.split("Already revealed")[1]


class TestParse:
    def test_accepts_a_legal_clue(self):
        assert parse_clue('thinking... {"clue": "Ocean", "number": 2}', ["Fish", "Car"]) == ("ocean", 2)

    @pytest.mark.parametrize("text", ['{"clue": "fishing", "number": 2}',   # contains a board word
                                      '{"clue": "sea water", "number": 2}',  # two words
                                      '{"clue": "ocean", "number": 5}',      # above the cap
                                      '{"clue": "ocean", "number": 0}',
                                      'no json here'])
    def test_rejects(self, text):
        assert isinstance(parse_clue(text, ["Fish", "Car"]), str)


class TestAsk:
    def test_retries_with_the_reason_then_succeeds(self):
        b = _board()
        own = b.words_by_role(Role.OWN)[0]
        spy = _spy([f'{{"clue": "{own}", "number": 1}}', '{"clue": "zzyzx", "number": 1}'])
        clue, number = spy.give_clue(TurnContext(b, 0), None)
        assert (clue, number) == ("zzyzx", 1)
        assert "not allowed" in spy._client.prompts[1]

    def test_raises_instead_of_inventing_a_clue(self):
        spy = _spy(["nonsense"] * 4)
        with pytest.raises(RuntimeError):
            spy.give_clue(TurnContext(_board(), 0), None)

    def test_cached_answer_is_not_rebought(self, tmp_path):
        b = _board()
        first = _spy(['{"clue": "zzyzx", "number": 2}'], tmp_path)
        assert first.give_clue(TurnContext(b, 0), None) == ("zzyzx", 2)
        again = _spy([], tmp_path)                     # no answers left: a call would fail
        assert again.give_clue(TurnContext(b, 0), None) == ("zzyzx", 2)
