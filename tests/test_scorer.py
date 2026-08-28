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
        # k=1, n=2: k < n -> natural stop, 1 own word then a miss.
        assert m[1, 2] == pytest.approx(1 * OWN_REWARD + DEFAULT_MISS_PENALTY)

    def test_budget_exhausted_before_a_miss(self):
        m = reward_matrix()
        # k=3, n=1: k >= n -> budget (n=1 attempt) exhausted, all correct.
        assert m[3, 1] == pytest.approx(1 * OWN_REWARD)

    def test_censored_top_bucket_exact_when_n_equals_max_k(self):
        m = reward_matrix()
        # k=4 (>=4, censored), n=4: k >= n -> budget (n=4 attempts)
        # exhausted, all correct -- no miss is assumed, since n never
        # exceeds MAX_K a censored k always lands in this branch correctly.
        assert m[4, 4] == pytest.approx(4 * OWN_REWARD)

    def test_custom_miss_penalty_only_affects_natural_stop_cells(self):
        lenient = reward_matrix(miss_penalty=-1.0)
        # k >= n cells don't involve any penalty at all.
        assert lenient[3, 1] == reward_matrix()[3, 1]
        # k < n cells do.
        assert lenient[1, 2] == pytest.approx(1 * OWN_REWARD - 1.0)


class TestExpectedRewardAndBestN:
    def test_certain_k_equals_max_prefers_n_equals_max(self):
        # A clue the model is 100% sure gets k=4: since n never exceeds
        # MAX_K, a censored k=4 always lands in the budget-exhausted
        # branch (no assumed miss), so the highest n=4 banks the most
        # correct guesses and scores strictly highest.
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 4] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n[0] == 4
        assert score[0] == pytest.approx(4 * OWN_REWARD)

    def test_certain_immediate_miss_scores_the_same_for_every_n_above_zero(self):
        # k=0 certain: any n>=1 spends its one real attempt on a guaranteed
        # miss (same penalty regardless of how many further attempts were
        # budgeted, since the turn already ended), so n=1..4 all tie.
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 0] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n[0] == 1  # n=0 would score higher (see below), but is floored out
        assert score[0] == pytest.approx(DEFAULT_MISS_PENALTY)

    def test_n_zero_is_excluded_by_default_even_though_it_would_win(self):
        # A clue certain to be an immediate miss (k=0): n=0 means zero real
        # attempts, so it guarantees 0 reward -- strictly better than any
        # n>=1, which all spend their one guaranteed-wrong attempt for
        # DEFAULT_MISS_PENALTY. n=0 is floored out regardless.
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 0] = 1.0
        best_n, _ = expected_reward_and_best_n(probs)
        assert best_n[0] != 0
        assert best_n[0] >= 1

    def test_min_n_can_be_relaxed_back_to_zero(self):
        probs = np.zeros((1, N_K_CLASSES), dtype=np.float32)
        probs[0, 0] = 1.0
        best_n, score = expected_reward_and_best_n(probs, min_n=0)
        assert best_n[0] == 0
        assert score[0] == pytest.approx(0.0)

    def test_batched_over_many_clues_at_once(self):
        probs = np.zeros((3, N_K_CLASSES), dtype=np.float32)
        probs[0, 0] = 1.0
        probs[1, 2] = 1.0
        probs[2, 4] = 1.0
        best_n, score = expected_reward_and_best_n(probs)
        assert best_n.shape == (3,)
        assert score.shape == (3,)

    def test_lower_risk_aversion_can_change_best_n(self):
        # Split probability between stopping at k=1 and k=3 -- a cautious
        # (default) penalty should pull the choice toward the lower,
        # safer peak (n=1) while a lenient one is willing to reach for
        # the higher peak (n=3) since overshooting the k=1 mass costs less.
        probs = np.array([[0.0, 0.5, 0.0, 0.5, 0.0]], dtype=np.float32)
        cautious_n, _ = expected_reward_and_best_n(probs, miss_penalty=-10.0)
        lenient_n, _ = expected_reward_and_best_n(probs, miss_penalty=-1.0)
        assert lenient_n[0] >= cautious_n[0]
        assert cautious_n[0] == 1
        assert lenient_n[0] == 3


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
