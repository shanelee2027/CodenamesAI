import itertools

import numpy as np

from codenames.pl_reward import gain_and_penalty
from codenames.spymasters.pick_temperature_listener import tempered_gain_and_penalty


def brute_force(s_own, s_bad, costs, max_k, temps):
    """Walk every ordered path of picks explicitly."""
    words = [("own", i, s) for i, s in enumerate(s_own)] + [("bad", i, s) for i, s in enumerate(s_bad)]
    gain = np.zeros(max_k)
    pen = np.zeros(max_k)

    def walk(left, prob, j, got):
        if j > max_k:
            return
        tau = temps[min(j, len(temps)) - 1]
        e = np.array([np.exp(w[2] / tau) for w in left])
        p = e / e.sum()
        for n, w in enumerate(left):
            q = prob * p[n]
            if w[0] == "own":
                gain[j - 1:] += q            # this pick counts for every k >= j
                walk(left[:n] + left[n + 1:], q, j + 1, got + 1)
            else:
                pen[j - 1:] += q * costs[w[1]]

    walk(words, 1.0, 1, 0)
    return gain, pen


def test_matches_brute_force_with_temperatures():
    rng = np.random.default_rng(0)
    for _ in range(5):
        s_own, s_bad = rng.normal(0, 2, 4), rng.normal(0, 2, 3)
        costs = np.array([0.2, 1.0, 10.0])
        temps = (0.9, 1.3, 1.7)
        g, p = tempered_gain_and_penalty(s_own[None], s_bad[None], costs, 4, temps)
        bg, bp = brute_force(s_own, s_bad, costs, 4, temps)
        assert np.allclose(g[0], bg, atol=1e-5) and np.allclose(p[0], bp, atol=1e-5)


def test_equals_plackett_luce_at_temperature_one():
    rng = np.random.default_rng(1)
    s_own, s_bad = rng.normal(0, 2, (50, 8)), rng.normal(0, 2, (50, 16))
    costs = rng.choice([0.2, 1.0, 10.0], 16)
    g, p = tempered_gain_and_penalty(s_own, s_bad, costs, 4, (1.0,))
    g0, p0 = gain_and_penalty(s_own, s_bad, costs, 4, cells=4096)
    assert np.allclose(g, g0, atol=1e-3) and np.allclose(p, p0, atol=1e-3)


def test_outside_option_is_a_free_non_team_word():
    rng = np.random.default_rng(2)
    s_own, s_bad, s_out = rng.normal(0, 2, (3, 5)), rng.normal(0, 2, (3, 6)), rng.normal(0, 2, 3)
    costs = np.full(6, 1.0)
    g, p = tempered_gain_and_penalty(s_own, s_bad, costs, 3, (1.0, 1.2), s_out=s_out)
    g2, p2 = tempered_gain_and_penalty(s_own, np.c_[s_bad, s_out], np.r_[costs, 0.0], 3, (1.0, 1.2))
    assert np.allclose(g, g2) and np.allclose(p, p2)


def test_flatter_later_picks_lower_the_value_of_a_one_sided_clue():
    # One dominant own word, weak others: the "95 / 3 / 1 / 1 for 4" case.
    s_own = np.array([[6.0, 2.5, 1.5, 1.5]])
    s_bad = np.array([[0.5, 0.0, 0.0, -0.5]])
    costs = np.array([0.2, 0.2, 1.0, 10.0])
    g1, p1 = tempered_gain_and_penalty(s_own, s_bad, costs, 4, (1.0,))
    g2, p2 = tempered_gain_and_penalty(s_own, s_bad, costs, 4, (1.0, 1.1, 1.25, 1.45))
    assert (g2 - p2)[0, 3] < (g1 - p1)[0, 3]
    assert np.isclose((g2 - p2)[0, 0], (g1 - p1)[0, 0])    # pick 1 unchanged
