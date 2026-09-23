"""Soft-label distillation (codenames/listener_training.py): a local model's
full distribution in place of the one word a teacher happened to pick."""

from __future__ import annotations

import numpy as np
import pytest

from codenames.listener_training import build_groups, group_log_loss, load_soft_labels
from codenames.local_lm import DistributionStore, StepDistribution, answer_prefix


def _position(soft=None):
    x = np.arange(12, dtype=float).reshape(4, 3)
    p = {"seed": 1, "clue": "c", "k": 2, "x": x, "n": 4, "targets": [2, 0]}
    if soft is not None:
        p["soft"] = soft
    return p


class TestGroups:
    def test_one_hot_when_there_is_no_soft_label(self):
        _, y, groups, _ = build_groups([_position()])
        assert groups == [4, 3]
        assert y[:4].tolist() == [0, 0, 1, 0] and y[4:].tolist() == [1, 0, 0]

    def test_soft_labels_follow_the_teachers_removals(self):
        """Step 2 drops the TEACHER's first pick (row 2) and renormalises the
        local model's distribution over what is left."""
        soft = [np.array([0.1, 0.2, 0.6, 0.1]), np.array([0.5, 0.25, 0.0, 0.25])]
        _, y, groups, _ = build_groups([_position(soft)])
        assert y[:4] == pytest.approx([0.1, 0.2, 0.6, 0.1])
        assert y[4:] == pytest.approx([0.5, 0.25, 0.25])


class TestLogLoss:
    def test_one_hot_cross_entropy_is_the_old_negative_log_likelihood(self):
        preds = np.array([1.0, 2.0, 0.5])
        ll, null = group_log_loss(preds, [3], np.array([0.0, 1.0, 0.0]))
        p = np.exp(preds) / np.exp(preds).sum()
        assert ll[0] == pytest.approx(-np.log(p[1]))
        assert null[0] == pytest.approx(np.log(3))

    def test_soft_cross_entropy_is_the_expectation(self):
        preds = np.array([1.0, 2.0, 0.5])
        y = np.array([0.2, 0.5, 0.3])
        logp = preds - np.log(np.exp(preds).sum())
        assert group_log_loss(preds, [3], y)[0][0] == pytest.approx(-(y * logp).sum())


class TestLoadSoftLabels:
    def _store(self, tmp_path):
        st = DistributionStore(tmp_path / "d.db")
        lp = np.log(np.array([0.7, 0.2, 0.1]))
        st.put_many("lm", "own", [("clue", ["a", "b", "c"], 2,
                    [StepDistribution(["a", "b", "c"], lp), StepDistribution(["b", "c"], np.log([0.9, 0.1]))])])
        return tmp_path / "d.db"

    def test_sources_are_kept_apart(self, tmp_path):
        """Distributions conditioned on another guesser's picks evaluate; they
        must never be loaded as training labels."""
        path = self._store(tmp_path)
        assert load_soft_labels(path, "lm", 1.0, source="some-guesser") == {}

    def test_temperature_one_is_the_model(self, tmp_path):
        d = load_soft_labels(self._store(tmp_path), "lm", 1.0)[("clue", ("a", "b", "c"), 2)]
        assert d[0]["a"] == pytest.approx(0.7, abs=1e-4) and d[1]["b"] == pytest.approx(0.9, abs=1e-4)

    def test_higher_temperature_softens(self, tmp_path):
        d = load_soft_labels(self._store(tmp_path), "lm", 3.0)[("clue", ("a", "b", "c"), 2)]
        assert 1 / 3 < d[0]["a"] < 0.7

    def test_temperature_zero_is_the_hard_label_control(self, tmp_path):
        d = load_soft_labels(self._store(tmp_path), "lm", 0)[("clue", ("a", "b", "c"), 2)]
        assert d[0] == {"a": 1.0, "b": 0.0, "c": 0.0}

    def test_hard_label_follows_the_scored_path_through_a_tie(self, tmp_path):
        """bf16 logits tie. The scorer took "b" at a tie with "a"; the hard label
        must be "b" too, or step 2's label would sit on a word already removed."""
        st = DistributionStore(tmp_path / "d.db")
        tie = np.log(np.array([0.45, 0.45, 0.1]))
        st.put_many("lm", "own", [("clue", ["a", "b", "c"], 2,
                    [StepDistribution(["a", "b", "c"], tie), StepDistribution(["a", "c"], np.log([0.9, 0.1]))])])
        d = load_soft_labels(tmp_path / "d.db", "lm", 0)[("clue", ("a", "b", "c"), 2)]
        assert d[0] == {"a": 0.0, "b": 1.0, "c": 0.0} and d[1] == {"a": 1.0, "c": 0.0}


def test_answer_prefix_matches_the_prompts_json_format():
    assert answer_prefix([]) == '["'
    assert answer_prefix(["Wave", "Beach"]) == '["Wave", "Beach", "'


class TestBoardIndex:
    def test_indexed_lookup_agrees_with_a_scan(self):
        from codenames.listener_training import BoardIndex, resolve_seed
        boards = [(1, frozenset({"a", "b", "c"})), (2, frozenset({"a", "b", "d"})), (3, frozenset({"x", "y", "z"}))]
        idx = BoardIndex(boards)
        for cands in (["a", "b"], ["A", "c"], ["y", "z"], ["a", "x"], ["q"]):
            assert resolve_seed(cands, idx) == resolve_seed(cands, boards)
        assert resolve_seed(["a", "b"], idx) is None, "two boards fit: ambiguous, never guessed"
