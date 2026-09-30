"""The win-probability objective (codenames/win_value.py) against brute-force
enumeration of the guesser's pick sequences."""

from __future__ import annotations

import numpy as np

from codenames.game import Role
from codenames.sequential_listener import SequentialParams, sequential_gain_and_penalty
from codenames.win_value import WinValue, win_probability


def _value(V: np.ndarray) -> WinValue:
    v = WinValue.__new__(WinValue)
    v.V = V
    return v


def _exact_gp(s_own, s_bad):
    """Exact Plackett-Luce on frozen scores: the sequential DP with every
    correction off."""
    def gp(costs):
        return sequential_gain_and_penalty(s_own, s_bad, costs, s_own.shape[1], np.zeros((10, 10)),
                                           SequentialParams.identity())
    return gp


def _enumerate_prefixes(s_own, s_bad, roles, k, V):
    """Exact P(win): a recursion over the words picked so far."""
    n_own = len(s_own)
    b = sum(r == Role.OPPONENT for r in roles)
    words = [("own", s) for s in s_own] + [(r.value, s) for r, s in zip(roles, s_bad)]

    def rec(left: tuple[int, ...], found: int) -> float:
        z = np.array([words[x][1] for x in left])
        probs = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        out = 0.0
        for pr, i in zip(probs, left):
            role = words[i][0]
            if role == "own":
                f = found + 1
                if f == n_own:
                    v = 1.0
                elif f == k:
                    v = 1 - V[b, n_own - f]
                else:
                    v = rec(tuple(x for x in left if x != i), f)
            else:
                a_left = n_own - found
                v = {"neutral": 1 - V[b, a_left], "assassin": 0.0,
                     "opponent": 0.0 if b == 1 else 1 - V[b - 1, a_left]}[role]
            out += pr * v
        return out

    return rec(tuple(range(len(words))), 0)


def test_matches_exact_enumeration():
    rng = np.random.default_rng(0)
    V = rng.uniform(0, 1, (10, 10))
    s_own = rng.normal(0, 1.5, (1, 3))
    roles = [Role.NEUTRAL, Role.OPPONENT, Role.OPPONENT, Role.ASSASSIN]
    s_bad = rng.normal(-0.5, 1.5, (1, 4))
    got = win_probability(_exact_gp(s_own, s_bad), roles, 3, 3, 2, _value(V))
    for k in (1, 2, 3):
        want = _enumerate_prefixes(s_own[0], s_bad[0], roles, k, V)
        assert abs(got[0, k - 1] - want) < 1e-6, (k, got[0, k - 1], want)


def test_last_opponent_word_loses():
    """With the opponent on its last word, hitting it ends the game."""
    rng = np.random.default_rng(1)
    V = rng.uniform(0, 1, (10, 10))
    s_own = rng.normal(0, 1, (1, 2))
    roles = [Role.NEUTRAL, Role.OPPONENT]
    s_bad = rng.normal(0, 1, (1, 2))
    got = win_probability(_exact_gp(s_own, s_bad), roles, 2, 2, 1, _value(V))
    for k in (1, 2):
        assert abs(got[0, k - 1] - _enumerate_prefixes(s_own[0], s_bad[0], roles, k, V)) < 1e-6


def test_only_the_assassin_matters_when_every_position_is_won():
    """With V = 0 everywhere (the opponent never wins from any position) and
    more than one opponent word left, P(win) = 1 - P(assassin before k)."""
    rng = np.random.default_rng(2)
    s_own = rng.normal(0, 1, (5, 3))
    roles = [Role.NEUTRAL, Role.OPPONENT, Role.OPPONENT, Role.ASSASSIN]
    s_bad = rng.normal(0, 1, (5, 4))
    gp = _exact_gp(s_own, s_bad)
    got = win_probability(gp, roles, 3, 3, 2, _value(np.zeros((10, 10))))
    _, p_ass = gp(np.array([0, 0, 0, 1.0]))
    np.testing.assert_allclose(got, 1 - p_ass, atol=1e-6)


def test_one_word_is_worth_the_value_gap():
    V = np.tile(np.linspace(0, 1, 10), (10, 1))          # V[x, y] rises with y
    v = _value(V)
    assert abs(v.one_word(3, 4) - (V[4, 3] - V[4, 2])) < 1e-12
