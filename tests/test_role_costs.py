"""codenames/game.py::role_costs -- a spymaster's risk appetite, as a
parameter rather than a global constant.

The point of the split is that `ROLE_REWARD` is the scoreboard and
`role_costs` is an opinion about it. Sweeping a spymaster's costs must not
move the yardstick every recorded result is measured in, so the two have to
be separable; the regression tests here are what keep them that way.

Reuses tests/test_expected_words_spymaster.py's synthetic fixtures (mean=0,
std=1, so the stored tensor value *is* the z-score) rather than rebuilding
them, since the question here is about the costs, not the metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from codenames.board import Role
from codenames.game import ROLE_REWARD, role_costs
from codenames.spymasters.expected_words import ExpectedWordsSpymaster
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

from tests.test_expected_words_spymaster import build, make_ctx, make_board, make_row, make_stats

NON_OWN = (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)


class TestRoleCosts:
    def test_defaults_are_the_games_own_reward_magnitudes(self):
        assert role_costs() == {r: abs(ROLE_REWARD[r]) for r in NON_OWN}

    def test_an_override_changes_only_the_role_it_names(self):
        costs = role_costs(opponent=2.5)
        assert costs[Role.OPPONENT] == 2.5
        assert costs[Role.NEUTRAL] == abs(ROLE_REWARD[Role.NEUTRAL])
        assert costs[Role.ASSASSIN] == abs(ROLE_REWARD[Role.ASSASSIN])

    def test_costs_are_positive_magnitudes_not_signed_rewards(self):
        """Every caller took abs() of the reward, so the parameter is the
        magnitude. A caller passing a negative number means it literally."""
        assert all(v > 0 for v in role_costs().values())
        assert role_costs(opponent=3.0)[Role.OPPONENT] == 3.0

    def test_does_not_mutate_role_reward(self):
        before = dict(ROLE_REWARD)
        role_costs(neutral=9.0, opponent=9.0, assassin=9.0)
        assert ROLE_REWARD == before


class TestSpymastersCarryTheirCosts:
    def test_expected_words_defaults_to_the_scoreboard(self):
        """Regression guard: every result on record was produced with these
        numbers, so the no-argument construction must not have moved."""
        sm = ExpectedWordsSpymaster(clue_stats=make_stats(["a"]))
        assert sm.costs == {r: abs(ROLE_REWARD[r]) for r in NON_OWN}

    def test_learned_listener_defaults_to_the_scoreboard(self):
        sm = LearnedListenerSpymaster(bundle=object(), clue_stats=make_stats(["a"]))
        assert sm.costs == {r: abs(ROLE_REWARD[r]) for r in NON_OWN}

    def test_learned_listener_passes_its_costs_to_the_shortlisting_stage(self):
        """The second stage can only rerank what the first hands it, so a
        cost that reached only stage two would be judged on a shortlist built
        under different beliefs."""
        sm = LearnedListenerSpymaster(
            bundle=object(), clue_stats=make_stats(["a"]),
            neutral_cost=0.5, opponent_cost=2.5, assassin_cost=25.0,
        )
        assert sm.costs == sm._first_stage.costs
        assert sm.costs[Role.OPPONENT] == 2.5


class TestCostsChangeTheChoice:
    """Behavioural: the parameter has to reach the score, not just be stored."""

    def _rows(self):
        # One clue pointing at two own words, with one opponent word sitting
        # between them -- so raising the opponent's price should make the
        # second own word not worth announcing.
        return {
            "clue": make_row(own=[2.0, 1.0] + [-20.0] * 7, opponent=[1.5] + [-100.0] * 7),
            "safe": make_row(own=[0.5] + [-20.0] * 8),
        }

    def test_a_dearer_opponent_word_lowers_the_score(self, tmp_path):
        cheap, sims = build(tmp_path, self._rows(), opponent_cost=1.0)
        dear, _ = build(tmp_path, self._rows(), opponent_cost=4.0)
        ctx = make_ctx(make_board())
        s_cheap = cheap._score_all_clues(ctx.board, sims)[1][0]
        s_dear = dear._score_all_clues(ctx.board, sims)[1][0]
        assert s_dear < s_cheap

    def test_a_dearer_opponent_word_does_not_raise_the_number(self, tmp_path):
        cheap, sims = build(tmp_path, self._rows(), opponent_cost=1.0)
        dear, _ = build(tmp_path, self._rows(), opponent_cost=8.0)
        ctx = make_ctx(make_board())
        assert dear._score_all_clues(ctx.board, sims)[0][0] <= cheap._score_all_clues(ctx.board, sims)[0][0]

    def test_assassin_cost_is_what_makes_the_assassin_expensive(self, tmp_path):
        """With the assassin priced at a neutral's rate, a clue next to it
        must stop being penalised like an assassin clue."""
        rows = {"clue": make_row(own=[2.0] + [-20.0] * 8, assassin=[1.5])}
        normal, sims = build(tmp_path, rows)
        defanged, _ = build(tmp_path, rows, assassin_cost=0.2)
        ctx = make_ctx(make_board())
        assert (defanged._score_all_clues(ctx.board, sims)[1][0]
                > normal._score_all_clues(ctx.board, sims)[1][0])

    def test_equal_costs_make_the_roles_interchangeable(self, tmp_path):
        """A sanity check on the plumbing: if every non-own word costs the
        same, moving the risk between roles cannot change the score."""
        near_opp = {"clue": make_row(own=[2.0] + [-20.0] * 8, opponent=[1.0] + [-100.0] * 7)}
        near_neu = {"clue": make_row(own=[2.0] + [-20.0] * 8, neutral=[1.0] + [-100.0] * 6)}
        flat = dict(neutral_cost=1.0, opponent_cost=1.0, assassin_cost=1.0)
        a, sims_a = build(tmp_path, near_opp, **flat)
        b, sims_b = build(tmp_path, near_neu, **flat)
        ctx = make_ctx(make_board())
        assert a._score_all_clues(ctx.board, sims_a)[1][0] == pytest.approx(
            b._score_all_clues(ctx.board, sims_b)[1][0], rel=1e-5)
