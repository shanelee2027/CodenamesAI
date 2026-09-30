"""The within-turn reward (codenames/sequential_listener.py) against brute
force, and against today's reward when the correction is zero."""

from __future__ import annotations

import itertools

import numpy as np

from codenames.pl_reward import gain_and_penalty
from codenames.sequential_listener import SequentialParams, sequential_gain_and_penalty


def brute(s_own, s_bad, costs, k, sim, p):
    """Enumerate pick sequences of own words explicitly; a non-own pick ends
    the turn and costs its price."""
    n_own = len(s_own)
    s = np.concatenate([s_own, s_bad])
    gain = pen = 0.0

    def rec(picked, prob, j):
        nonlocal gain, pen
        if j > k:
            return
        left = [i for i in range(len(s)) if i not in picked]
        if picked:
            kk = min(j, 4) - 2
            alpha = np.exp(p.a[kk] + p.b[kk] * (max(s[left]) - max(s[list(picked)])))
            z = np.array([alpha * s[i] + p.beta[kk] * max(sim[i, o] for o in picked) for i in left])
        else:
            z = s[left]
        q = np.exp(z - z.max())
        q /= q.sum()
        for i, qi in zip(left, q):
            if i < n_own:
                gain += prob * qi
                rec(picked | {i}, prob * qi, j + 1)
            else:
                pen += prob * qi * costs[i - n_own]

    rec(frozenset(), 1.0, 1)
    return gain, pen


def test_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    n_own, n_bad = 4, 3
    sim = rng.uniform(-0.2, 0.8, (n_own + n_bad, n_own + n_bad))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0)
    p = SequentialParams(a=np.array([-0.3, -0.4, -0.6]), b=np.array([-0.1, 0.05, -0.07]),
                         beta=np.array([5.0, 4.0, 3.0]), gamma=np.zeros(3))
    s_own = rng.normal(0, 2, (3, n_own))
    s_bad = rng.normal(0, 2, (3, n_bad))
    costs = np.array([0.2, 1.0, 10.0])
    g, pe = sequential_gain_and_penalty(s_own, s_bad, costs, 4, sim, p)
    for c in range(3):
        for k in range(1, 5):
            bg, bp = brute(s_own[c], s_bad[c], costs, k, sim, p)
            assert abs(g[c, k - 1] - bg) < 1e-5 and abs(pe[c, k - 1] - bp) < 1e-5


def test_zero_correction_is_todays_reward() -> None:
    rng = np.random.default_rng(1)
    s_own, s_bad = rng.normal(0, 2, (5, 4)), rng.normal(0, 2, (5, 5))
    costs = np.array([0.2, 0.2, 1.0, 1.0, 10.0])
    sim = rng.uniform(0, 1, (9, 9))
    g, pe = sequential_gain_and_penalty(s_own, s_bad, costs, 4, sim, SequentialParams.identity())
    g0, p0 = gain_and_penalty(s_own, s_bad, costs, 4)
    assert np.allclose(g, g0, atol=2e-3) and np.allclose(pe, p0, atol=2e-3)


def test_similar_second_word_raises_the_value_of_two() -> None:
    s_own = np.array([[3.0, 1.0]])
    s_bad = np.array([[1.0, 1.0]])
    costs = np.array([1.0, 1.0])
    p = SequentialParams(a=np.zeros(3), b=np.zeros(3), beta=np.full(3, 5.0), gamma=np.zeros(3))
    far, near = np.zeros((4, 4)), np.zeros((4, 4))
    near[0, 1] = near[1, 0] = 0.8
    g_far, _ = sequential_gain_and_penalty(s_own, s_bad, costs, 2, far, p)
    g_near, _ = sequential_gain_and_penalty(s_own, s_bad, costs, 2, near, p)
    assert g_near[0, 1] > g_far[0, 1] and g_near[0, 0] == g_far[0, 0]
