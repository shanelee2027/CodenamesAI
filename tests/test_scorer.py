from __future__ import annotations

import numpy as np
import pytest
import torch

from codenames.board import Role
from codenames.scorer import (
    DEFAULT_MISS_PENALTY,
    N_OUTCOME_CLASSES,
    OWN_REWARD,
    LinearScorer,
    Scorer,
    decode_outcome_class,
    expected_reward_and_best_n,
    outcome_class,
    reward_matrix,
)

MAX_K = 4  # matches codenames.board.MAX_CLUE_NUMBER, mirrored here to keep test intent readable


class TestOutcomeClass:
    def test_k_zero_causes_pack_into_the_first_three_classes(self):
        assert outcome_class(0, Role.NEUTRAL) == 0
        assert outcome_class(0, Role.OPPONENT) == 1
        assert outcome_class(0, Role.ASSASSIN) == 2

    def test_higher_k_causes_pack_in_order(self):
        assert outcome_class(3, Role.NEUTRAL) == 9
        assert outcome_class(3, Role.OPPONENT) == 10
        assert outcome_class(3, Role.ASSASSIN) == 11

    def test_censored_bucket_is_the_last_class(self):
        assert outcome_class(MAX_K, None) == N_OUTCOME_CLASSES - 1

    def test_rejects_a_cause_for_the_censored_k(self):
        with pytest.raises(ValueError):
            outcome_class(MAX_K, Role.NEUTRAL)

    def test_rejects_a_missing_cause_below_max_k(self):
        with pytest.raises(ValueError):
            outcome_class(2, None)


class TestDecodeOutcomeClass:
    def test_round_trips_every_valid_class(self):
        for k in range(MAX_K):
            for cause in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN):
                cls = outcome_class(k, cause)
                assert decode_outcome_class(cls) == (k, cause)
        assert decode_outcome_class(outcome_class(MAX_K, None)) == (MAX_K, None)

    def test_total_class_count(self):
        assert N_OUTCOME_CLASSES == MAX_K * 3 + 1 == 13


class TestRewardMatrix:
    def test_shape(self):
        assert reward_matrix().shape == (N_OUTCOME_CLASSES, MAX_K + 1)

    def test_defaults_are_the_true_game_reward_table_not_baseline_3s(self):
        # Confirmed this session: neutral defaults to 0.0 (the real §2
        # reward), not SCOPE baseline-3's separate untuned -0.3 constant.
        m = reward_matrix()
        cls = outcome_class(0, Role.NEUTRAL)
        assert m[cls, 1] == pytest.approx(0.0)

    def test_natural_stop_charges_the_causes_own_value(self):
        m = reward_matrix()
        # k=1, n=2 (k < n -> natural stop, 1 own word then the miss).
        assert m[outcome_class(1, Role.NEUTRAL), 2] == pytest.approx(1 * OWN_REWARD + 0.0)
        assert m[outcome_class(1, Role.OPPONENT), 2] == pytest.approx(1 * OWN_REWARD - 1.0)
        assert m[outcome_class(1, Role.ASSASSIN), 2] == pytest.approx(1 * OWN_REWARD + DEFAULT_MISS_PENALTY)

    def test_budget_exhausted_ignores_cause(self):
        m = reward_matrix()
        # k=3, n=1 (k >= n -> budget exhausted, all n=1 correct) -- the
        # cause never even happened within budget, so it shouldn't matter.
        neutral = m[outcome_class(3, Role.NEUTRAL), 1]
        assassin = m[outcome_class(3, Role.ASSASSIN), 1]
        assert neutral == assassin == pytest.approx(1 * OWN_REWARD)

    def test_censored_class_is_always_budget_exhausted(self):
        m = reward_matrix()
        cls = outcome_class(MAX_K, None)
        for n in range(MAX_K + 1):
            assert m[cls, n] == pytest.approx(n * OWN_REWARD)

    def test_own_reward_scales_both_branches(self):
        m = reward_matrix(own_reward=2.0)
        assert m[outcome_class(3, Role.NEUTRAL), 1] == pytest.approx(1 * 2.0)  # budget branch
        assert m[outcome_class(1, Role.NEUTRAL), 2] == pytest.approx(1 * 2.0 + 0.0)  # natural branch

    def test_neutral_reward_only_affects_neutral_causes(self):
        m = reward_matrix(neutral_reward=-0.3)
        assert m[outcome_class(1, Role.NEUTRAL), 2] == pytest.approx(1 * OWN_REWARD - 0.3)
        assert m[outcome_class(1, Role.OPPONENT), 2] == reward_matrix()[outcome_class(1, Role.OPPONENT), 2]
        assert m[outcome_class(1, Role.ASSASSIN), 2] == reward_matrix()[outcome_class(1, Role.ASSASSIN), 2]

    def test_opponent_reward_only_affects_opponent_causes(self):
        m = reward_matrix(opponent_reward=-2.0)
        assert m[outcome_class(1, Role.OPPONENT), 2] == pytest.approx(1 * OWN_REWARD - 2.0)
        assert m[outcome_class(1, Role.NEUTRAL), 2] == reward_matrix()[outcome_class(1, Role.NEUTRAL), 2]

    def test_assassin_reward_only_affects_assassin_causes(self):
        lenient = reward_matrix(assassin_reward=-1.0)
        assert lenient[outcome_class(1, Role.ASSASSIN), 2] == pytest.approx(1 * OWN_REWARD - 1.0)
        assert lenient[outcome_class(1, Role.OPPONENT), 2] == reward_matrix()[outcome_class(1, Role.OPPONENT), 2]
        # And never affects any budget-exhausted cell, regardless of cause.
        assert lenient[outcome_class(3, Role.ASSASSIN), 1] == reward_matrix()[outcome_class(3, Role.ASSASSIN), 1]


class TestExpectedRewardAndBestN:
    def test_certain_k_equals_max_prefers_n_equals_max(self):
        # A clue the model is 100% sure gets k=4 (the censored class):
        # since n never exceeds MAX_K, this always lands in the
        # budget-exhausted branch (no assumed miss), so the highest n=4
        # banks the most correct guesses and scores strictly highest.
        probs = np.zeros((1, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(MAX_K, None)] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n[0] == 4
        assert score[0] == pytest.approx(4 * OWN_REWARD)

    def test_certain_immediate_assassin_scores_the_same_for_every_n_above_zero(self):
        # k=0/assassin certain: any n>=1 spends its one real attempt on a
        # guaranteed assassin hit (same penalty regardless of how many
        # further attempts were budgeted, since the turn already ended),
        # so n=1..4 all tie.
        probs = np.zeros((1, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(0, Role.ASSASSIN)] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n[0] == 1  # n=0 would score higher (see below), but is floored out
        assert score[0] == pytest.approx(DEFAULT_MISS_PENALTY)

    def test_n_zero_is_excluded_by_default_even_though_it_would_win(self):
        # n=0 means zero real attempts, so it guarantees 0 reward --
        # strictly better than any n>=1, which all spend their one
        # guaranteed-wrong attempt for DEFAULT_MISS_PENALTY. n=0 is
        # floored out regardless.
        probs = np.zeros((1, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(0, Role.ASSASSIN)] = 1.0
        best_n, _ = expected_reward_and_best_n(probs)
        assert best_n[0] != 0
        assert best_n[0] >= 1

    def test_min_n_can_be_relaxed_back_to_zero(self):
        probs = np.zeros((1, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(0, Role.ASSASSIN)] = 1.0
        best_n, score = expected_reward_and_best_n(probs, min_n=0)
        assert best_n[0] == 0
        assert score[0] == pytest.approx(0.0)

    def test_batched_over_many_clues_at_once(self):
        probs = np.zeros((3, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(0, Role.ASSASSIN)] = 1.0
        probs[1, outcome_class(2, Role.OPPONENT)] = 1.0
        probs[2, outcome_class(MAX_K, None)] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n.shape == (3,)
        assert score.shape == (3,)

    def test_lower_risk_aversion_can_change_best_n(self):
        # Split probability between stopping-on-assassin at k=1 and k=3 --
        # a cautious (default) assassin_reward should pull the choice
        # toward the lower, safer peak (n=1) while a lenient one is
        # willing to reach for the higher peak (n=3) since overshooting
        # the k=1 mass costs less.
        probs = np.zeros((1, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(1, Role.ASSASSIN)] = 0.5
        probs[0, outcome_class(3, Role.ASSASSIN)] = 0.5
        cautious_n, _ = expected_reward_and_best_n(probs, assassin_reward=-10.0)
        lenient_n, _ = expected_reward_and_best_n(probs, assassin_reward=-1.0)
        assert lenient_n[0] >= cautious_n[0]
        assert cautious_n[0] == 1
        assert lenient_n[0] == 3

    def test_neutral_and_opponent_rewards_are_independently_adjustable(self):
        # Same clue, certain to stop at k=0 on a neutral word -- a more
        # negative neutral_reward should make it look worse, independent
        # of assassin_reward/opponent_reward staying at their defaults.
        probs = np.zeros((1, N_OUTCOME_CLASSES), dtype=np.float32)
        probs[0, outcome_class(0, Role.NEUTRAL)] = 1.0
        _, lenient_score = expected_reward_and_best_n(probs, neutral_reward=0.0)
        _, harsh_score = expected_reward_and_best_n(probs, neutral_reward=-5.0)
        assert harsh_score[0] < lenient_score[0]


class TestScorer:
    def test_forward_shape(self):
        model = Scorer(input_dim=103)
        x = torch.randn(8, 103)
        logits = model(x)
        assert logits.shape == (8, N_OUTCOME_CLASSES)

    def test_predict_proba_sums_to_one(self):
        model = Scorer(input_dim=103)
        x = torch.randn(8, 103)
        probs = model.predict_proba(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


class TestLinearScorer:
    def test_forward_shape(self):
        model = LinearScorer(input_dim=103)
        x = torch.randn(8, 103)
        logits = model(x)
        assert logits.shape == (8, N_OUTCOME_CLASSES)

    def test_predict_proba_sums_to_one(self):
        model = LinearScorer(input_dim=103)
        x = torch.randn(8, 103)
        probs = model.predict_proba(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_has_no_hidden_layers(self):
        model = LinearScorer(input_dim=103)
        assert isinstance(model.net, torch.nn.Linear)
