"""The free-association Poisson term (codenames/listener_training.py): counts
matched to board words, aligned to build_groups rows, and a gradient that
really is the derivative of the loss it claims to minimise."""

from __future__ import annotations

import numpy as np
import pytest

from codenames.listener_training import (
    _assoc_norm, association_counts, association_targets, build_groups, fit_rate_link,
    group_softmax_objective, poisson_deviance, split_positions,
)


def _lists(*lists):
    return [{_assoc_norm(w) for w in lst} for lst in lists]


class TestCounts:
    def test_each_list_counts_once_and_plurals_match(self):
        lists = _lists(["bells", "church", "Bell"], ["ring", "church"], ["new-york"])
        assert association_counts(["Bell", "Church", "New York", "Ring", "Fish"], lists).tolist() \
            == [1, 2, 1, 1, 0]

    def test_non_ascii_hyphen_is_stripped(self):
        assert association_counts(["Ice Cream"], _lists(["ice‑cream"])).tolist() == [1]


class TestTargets:
    def _pos(self, clue="c", k=2, n=4, decoy=False):
        return {"seed": 1, "clue": clue, "k": k, "n": n, "x": np.zeros((n, 1)), "targets": [0, 1],
                "words": ["A", "B", "C", "D"][:n], **({"decoy": True} if decoy else {})}

    def test_only_step_one_rows_carry_a_count(self):
        y, r = association_targets([self._pos()], {"c": _lists(["a", "b"], ["a"])})
        _, _, groups, _ = build_groups([self._pos()])
        assert len(y) == len(r) == sum(groups) == 7
        assert y.tolist() == [2, 1, 0, 0, 0, 0, 0]
        assert r.tolist() == [2, 2, 2, 2, 0, 0, 0]

    def test_positions_without_lists_and_decoy_boards_are_excluded(self):
        y, r = association_targets([self._pos(clue="other"), self._pos(decoy=True)],
                                   {"c": _lists(["a"])})
        assert not r.any() and not y.any()


class TestRateLink:
    def test_recovers_slope_and_offset(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=20000)
        r = np.full_like(s, 5.0)
        y = rng.poisson(r * np.exp(0.7 * s - 2.0)).astype(float)
        a, b = fit_rate_link(s, y, r)
        assert a == pytest.approx(0.7, abs=0.03) and b == pytest.approx(-2.0, abs=0.03)

    def test_deviance_is_zero_at_the_counts(self):
        y = np.array([0.0, 1.0, 3.0])
        assert poisson_deviance(y, np.array([1e-12, 1.0, 3.0])) == pytest.approx(0.0, abs=1e-9)


class TestObjective:
    def test_gradient_matches_finite_differences_of_the_joint_loss(self):
        groups = [3, 2]
        lab = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        ay = np.array([2.0, 0.0, 1.0, 0.0, 0.0])
        ar = np.array([3.0, 3.0, 3.0, 0.0, 0.0])
        lam, a, b = 0.5, 0.8, -1.0

        def loss(s):
            out, st = 0.0, 0
            for g in groups:
                z = s[st:st + g]
                out -= np.dot(lab[st:st + g], z - np.log(np.exp(z).sum()))
                st += g
            mu = ar * np.exp(a * s + b)
            return out + lam * np.sum(mu - ay * (a * s + b))

        class _D:
            def get_label(self):
                return lab

        s = np.array([0.3, -0.2, 1.1, 0.0, 0.4])
        grad, hess = group_softmax_objective(groups, assoc=(ay, ar, lam, a, b))(s, _D())
        eps = 1e-6
        num = np.array([(loss(s + eps * np.eye(5)[i]) - loss(s - eps * np.eye(5)[i])) / (2 * eps)
                        for i in range(5)])
        assert grad == pytest.approx(num, abs=1e-5)
        assert (hess > 0).all()


def test_split_is_the_one_train_listener_always_drew():
    """Same rng calls in the same order as the inline split it replaced."""
    pos = [{"seed": s} for s in range(100)] + [{"seed": 1000 + s, "decoy": True} for s in range(20)]
    tr, va = split_positions(pos, 0.25, 0)
    rng = np.random.default_rng(0)
    want = set(rng.choice(list(range(100)), size=25, replace=False).tolist())
    want |= set(np.random.default_rng(1).choice(list(range(1000, 1020)), size=5, replace=False).tolist())
    assert {p["seed"] for p in va} == want
    assert len(tr) + len(va) == 120
