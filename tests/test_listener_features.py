"""Guards on the distilled listener's feature extraction and evaluation.

Two of these exist because the bug they catch is silent: it produces a
plausible number rather than an error.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from codenames.listener_features import FEATURE_NAMES, N_FEATURES, extract, p_is_max


class TestNoRoleLeakage:
    def test_extract_cannot_see_roles(self):
        """The listener is shown a clue and a word list, never which words are
        own/opponent/neutral/assassin. A role-aware feature would make the
        distilled guesser look excellent in the arena and worthless as a model
        of a listener, without failing anything."""
        params = set(inspect.signature(extract).parameters)
        assert not {"board", "roles", "role_of", "own", "opponent", "assassin"} & params
        assert not any("role" in n or "own_word" in n or "assassin" in n for n in FEATURE_NAMES)

    def test_feature_names_match_width(self):
        assert len(FEATURE_NAMES) == N_FEATURES == len(set(FEATURE_NAMES))


class TestPIsMax:
    def test_probabilities_sum_to_one(self):
        z = np.array([3.0, 1.0, 0.5, -2.0])
        for sigma in (0.5, 2.0, 5.0):
            assert p_is_max(z, sigma).sum() == pytest.approx(1.0, abs=2e-3)

    def test_best_word_is_most_likely(self):
        z = np.array([3.0, 1.0, 0.5, -2.0])
        assert int(np.argmax(p_is_max(z, 2.0))) == 0

    def test_large_sigma_approaches_uniform(self):
        z = np.array([3.0, 1.0, 0.5, -2.0])
        p = p_is_max(z, 50.0)
        assert p.max() - p.min() < 0.05

    def test_small_sigma_approaches_certainty(self):
        z = np.array([3.0, 1.0, 0.5, -2.0])
        assert p_is_max(z, 0.05)[0] > 0.99


class TestTieAwareAccuracy:
    """The target sits at index 0 of every group, and np.argmax returns the
    FIRST maximum -- so a constant-scoring model would grade 100% correct, and
    the more heavily it was regularised the better it would look. This
    actually happened; see docs/log.md."""

    def _acc(self, preds, groups, y):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pipeline"))
        from train_listener import accuracy_on
        return accuracy_on(np.asarray(preds, dtype=float), groups, np.asarray(y, dtype=float))

    def test_constant_model_scores_chance_not_one(self):
        groups = [4, 4]
        y = [1, 0, 0, 0, 1, 0, 0, 0]
        assert self._acc([0.0] * 8, groups, y) == pytest.approx(0.25)

    def test_correct_model_scores_one(self):
        groups = [4]
        assert self._acc([5.0, 1.0, 0.0, -1.0], groups, [1, 0, 0, 0]) == pytest.approx(1.0)

    def test_wrong_model_scores_zero(self):
        groups = [4]
        assert self._acc([0.0, 9.0, 1.0, -1.0], groups, [1, 0, 0, 0]) == pytest.approx(0.0)

    def test_two_way_tie_at_the_top_scores_half(self):
        groups = [4]
        assert self._acc([5.0, 5.0, 0.0, -1.0], groups, [1, 0, 0, 0]) == pytest.approx(0.5)
