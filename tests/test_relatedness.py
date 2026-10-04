import pytest

from codenames.relatedness import RelatednessStore, parse

WORDS = ["Alien", "Triangle", "Spot", "Sound", "Park"]


def test_parse_returns_board_order_and_drops_unknown_names():
    text = '{"guess": ["Spot", "alien", "Moon"], "stretch": ["Park", "Spot"]}'
    guess, stretch, unknown = parse(text, WORDS)
    assert guess == ["Alien", "Spot"]
    assert stretch == ["Park"]                     # Spot is already a guess
    assert unknown == 1


def test_parse_takes_the_last_answer_and_allows_empty_lists():
    text = 'Draft: {"guess": ["Park"], "stretch": []}\nFinal: {"guess": [], "stretch": ["Sound"]}'
    assert parse(text, WORDS)[:2] == ([], ["Sound"])


def test_parse_rejects_a_response_without_an_answer():
    with pytest.raises(ValueError):
        parse('["Alien", "Spot"]', WORDS)


def test_store_round_trip(tmp_path):
    s = RelatednessStore(tmp_path / "x.db")
    assert s.get("m", "obscure", WORDS) is None
    s.put("m", "obscure", WORDS, ["Alien"], ["Spot"], "raw")
    assert s.get("m", "obscure", WORDS) == (["Alien"], ["Spot"])
    assert s.get("m", "obscure", WORDS[::-1]) is None          # the candidate order is part of the key
