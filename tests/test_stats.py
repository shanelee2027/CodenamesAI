"""codenames/stats.py and codenames/headtohead.py: the statistics every
reported comparison rests on."""

from __future__ import annotations

import json

import pytest

from codenames.headtohead import paired_summary, side_turn_stats
from codenames.stats import binom_two_sided, fisher_2x2, wilson


class TestStats:
    def test_sign_test_matches_known_values(self):
        assert binom_two_sided(5, 10) == pytest.approx(1.0)
        assert binom_two_sided(0, 10) == pytest.approx(2 / 1024)
        assert binom_two_sided(0, 0) == 1.0

    def test_wilson_brackets_the_proportion(self):
        lo, hi = wilson(30, 100)
        assert lo < 0.30 < hi
        assert wilson(0, 0) == (0.0, 1.0)

    def test_fisher_is_one_for_identical_rows(self):
        assert fisher_2x2(5, 5, 5, 5) == pytest.approx(1.0)


def _row(seed, a, b, winner, turns=()):
    return (f"run|A={a},B={b}", seed, winner, json.dumps(list(turns)))


class TestPairedSummary:
    def test_sweeps_splits_and_unpaired_boards(self):
        rows = [
            _row(1, "x", "y", "A"), _row(1, "y", "x", "B"),   # x sweeps board 1
            _row(2, "x", "y", "A"), _row(2, "y", "x", "A"),   # split
            _row(3, "x", "y", "A"),                           # half-played: ignored
        ]
        s = paired_summary(rows)
        assert s.boards == 2 and s.games == 4
        assert dict(s.swept) == {"x": 1} and s.split == 1
        assert s.wins["x"] == 3 and s.wins["y"] == 1
        assert s.sign_p("x") == pytest.approx(1.0)

    def test_a_timeout_is_a_win_for_nobody(self):
        s = paired_summary([_row(1, "x", "y", None), _row(1, "y", "x", None)])
        assert sum(s.wins.values()) == 0 and s.split == 1

    def test_turns_are_credited_to_the_side_that_took_them(self):
        turns = [{"team": "A", "number": 2, "guesses": [["w", "own"], ["v", "neutral"]]},
                 {"team": "B", "number": 1, "guesses": [["u", "own"]]}]
        st = side_turn_stats([_row(1, "x", "y", "A", turns)])
        assert st["x"].mean_k == 2 and st["x"].own_per_clue == 1 and st["x"].own_rate == 0.5
        assert st["y"].own_per_clue == 1
