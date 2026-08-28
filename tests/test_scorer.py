from __future__ import annotations

import numpy as np
import pytest
import torch

from codenames.scorer import (
    DEFAULT_MISS_PENALTY,
    N_K_CLASSES,
    OWN_REWARD,
    LinearScorer,
    Scorer,
    expected_reward_and_best_n,
    reward_matrix,
)


class TestRewardMatrix:
    def test_shape(self):
        assert reward_matrix().shape == (N_K_CLASSES, N_K_CLASSES)

    def test_natural_stop_within_budget(self):
        m = reward_matrix()
        # k=1, n=2: k <= n -> natural stop, 1 own word then a miss.
        assert m[1, 2] == pytest.approx(1 * OWN_REWARD + DEFAULT_MISS_PENALTY)

    def test_budget_exhausted_before_a_miss(self):
        m = reward_matrix()
        # k=3, n=1: k > n -> budget (n+1=2 attempts) exhausted, all correct.
        assert m[3, 1] == pytest.approx(2 * OWN_REWARD)

    def test_censored_top_bucket_treated_as_exact_when_n_equals_max_k(self):
        m = reward_matrix()
        # k=4 (>=4, censored), n=4: treated as if k==4 exactly -> a miss is
        # assumed at the 5th attempt, per the module's documented approximation.
        assert m[4, 4] == pytest.approx(4 * OWN_REWARD + DEFAULT_MISS_PENALTY)

    def test_custom_miss_penalty_only_affects_natural_stop_cells(self):
        lenient = reward_matrix(miss_penalty=-1.0)
        # k > n cells don't involve any penalty at all.
        assert lenient[3, 1] == reward_matrix()[3, 1]
        # k <= n cells do.
        assert lenient[1, 2] == pytest.approx(1 * OWN_REWARD - 1.0)


class TestExpectedRewardAndBestN:
    def test_certain_k_equals_max_prefers_n_one_below_max(self):
        # A clue the model is 100% sure gets k=4: taking n=4 assumes a miss
        # at attempt 5 (the censored-bucket approximation), so n=3 (banking
        # exactly 4 correct with no assumed miss) scores strictly higher --
        # this is the documented tradeoff of treating k=4 as exact.
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 4] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n[0] == 3
        assert score[0] == pytest.approx(4 * OWN_REWARD)

    def test_certain_immediate_miss_scores_the_same_for_every_n(self):
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 0] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n[0] == 1  # n=0 ties too, but is excluded by the default floor
        assert score[0] == pytest.approx(DEFAULT_MISS_PENALTY)

    def test_n_zero_is_excluded_by_default_even_though_it_would_win(self):
        # A clue certain to get k=1 right: n=0 scores strictly *higher*
        # than n=1 here (budget-exhausted-with-1-correct vs. a real
        # decision at n=1 that still risks nothing since k=1, but n=0 is
        # floored out regardless of whether it would have won).
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 1] = 1.0
        best_n, _ = expected_reward_and_best_n(probs)
        assert best_n[0] != 0
        assert best_n[0] >= 1

    def test_min_n_can_be_relaxed_back_to_zero(self):
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 1] = 1.0
        best_n, _ = expected_reward_and_best_n(probs, min_n=0)
        assert best_n[0] == 0

    def test_batched_over_many_clues_at_once(self):
        probs = np.zeros((3, N_K_CLASSES), dtype=np.float32)
        probs[0, 0] = 1.0
        probs[1, 2] = 1.0
        probs[2, 4] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n.shape == (3,)
        assert score.shape == (3,)

    def test_lower_risk_aversion_can_change_best_n(self):
        # Split probability between an immediate miss and a big win --
        # a cautious (default) penalty should discourage a high n more
        # than a lenient one.
        probs = np.array([[0.5, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        cautious_n, _ = expected_reward_and_best_n(probs, miss_penalty=-10.0)
        lenient_n, _ = expected_reward_and_best_n(probs, miss_penalty=-1.0)
        assert lenient_n[0] >= cautious_n[0]


class TestScorer:
    def test_forward_shape(self):
        model = Scorer(input_dim=103)
        x = torch.randn(8, 103)
        logits = model(x)
        assert logits.shape == (8, N_K_CLASSES)

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
        assert logits.shape == (8, N_K_CLASSES)

    def test_predict_proba_sums_to_one(self):
        model = LinearScorer(input_dim=103)
        x = torch.randn(8, 103)
        probs = model.predict_proba(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_has_no_hidden_layers(self):
        model = LinearScorer(input_dim=103)
        assert isinstance(model.net, torch.nn.Linear)
