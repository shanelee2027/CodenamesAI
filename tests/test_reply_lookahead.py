"""reply_lookahead_listener's turn enumeration agrees with the frozen
Plackett-Luce turn model the search uses (codenames/pl_reward.py)."""

import numpy as np

from codenames.pl_reward import gain_and_penalty
from codenames.spymasters.reply_lookahead_listener import turn_outcomes


def test_enumeration_matches_pl_reward():
    rng = np.random.default_rng(0)
    n_own, n_bad = 4, 6
    s = rng.normal(0, 1.5, n_own + n_bad)
    for k in (1, 2, 3):
        assert abs(sum(turn_outcomes(s, n_own, k).values()) - 1.0) < 5e-3     # pruning drops little
        out = turn_outcomes(s, n_own, k, prune=0.0)
        assert abs(sum(out.values()) - 1.0) < 1e-9
        # P(the guesser gets its j-th own word) and P(j-1 own words, then bad word w).
        reach = [sum(p for (f, _), p in out.items() if len(f) >= j) for j in range(1, k + 1)]
        costs = np.eye(n_bad)
        for w in range(n_bad):
            gain, pen = gain_and_penalty(s[None, :n_own], s[None, n_own:], costs[w], k, cells=4000)
            ends = [sum(p for (f, e), p in out.items() if e == n_own + w and len(f) == j) for j in range(k)]
            assert np.allclose(np.cumsum(ends), pen[0], atol=2e-3)
        assert np.allclose(np.cumsum(reach), gain[0], atol=2e-3)


def test_turn_stops_at_the_number_and_at_the_last_own_word():
    s = np.array([5.0, 5.0, -5.0])                                  # two own words, both near-certain
    out = turn_outcomes(s, 2, 4)
    assert max(len(f) for f, _ in out) == 2
    assert all(e is None for f, e in out if len(f) == 2)
