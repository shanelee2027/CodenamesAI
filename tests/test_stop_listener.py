"""stop_listener's turn value: without STOP it is pick_index_lookahead's exact
enumeration, and STOP ends the turn like a neutral word."""

import numpy as np
import pytest

from codenames.game import Role
from codenames.spymasters.pick_index_lookahead_listener import (PickIndexLookaheadListenerSpymaster,
                                                                pick_turn_outcomes)
from codenames.spymasters.stop_listener import turn_values


class _Value:
    V = np.random.default_rng(1).uniform(0.2, 0.8, (10, 10))


ROLES = [Role.NEUTRAL, Role.NEUTRAL, Role.OPPONENT, Role.OPPONENT, Role.ASSASSIN]


def test_without_stop_it_is_the_pick_index_enumeration():
    rng = np.random.default_rng(0)
    n_own, K = 3, 3
    S = rng.normal(0, 1.5, (K, n_own + len(ROLES)))
    sm = PickIndexLookaheadListenerSpymaster.__new__(PickIndexLookaheadListenerSpymaster)
    sm.value = _Value()
    got = turn_values(np.hstack([S, np.full((K, 1), -np.inf)]), n_own, ROLES, _Value.V, K)
    for k in range(1, K + 1):
        want = sm.turn_value(pick_turn_outcomes(S[:k], n_own, k, prune=0.0), n_own, ROLES)
        assert got[k - 1] == pytest.approx(want)


def test_stop_ends_the_turn_like_a_neutral_word():
    # Two own words. Pick 1 is own word 0 for sure; at pick 2 the guesser
    # stops for sure, so any number above 1 is worth exactly a clue for 1.
    S = np.full((3, 2 + len(ROLES) + 1), -np.inf)
    S[0, 0] = 0.0
    S[1:, -1] = 0.0
    v = turn_values(S, 2, ROLES, _Value.V, 3)
    one_left = 1 - _Value.V[2, 1]                   # the opponent, with 2 words, to move; we have 1
    assert v == pytest.approx([one_left] * 3)


def test_opponent_and_assassin_endings():
    # One own word; pick 1 is the assassin, an opponent word or the own word, a third each.
    S = np.full((1, 1 + len(ROLES) + 1), -np.inf)
    S[0, [0, 3, 5]] = 0.0                           # own, the first opponent word, the assassin
    v = turn_values(S, 1, ROLES, _Value.V, 1)
    assert v[0] == pytest.approx((1.0 + (1 - _Value.V[1, 1]) + 0.0) / 3)


def test_builds_from_the_registry_with_its_own_files():
    from codenames.spymasters.registry import spymaster_spec

    cls, kw = spymaster_spec("stop_listener")
    files = {p.name for p in cls.model_files(kw)}
    assert {"listener_gbt_assoc_profile.txt", "listener_gbt_stop.txt", "assoc_profile.npz",
            "win_value.npz"} <= files


def test_points_at_counts_own_picks_until_stop_and_skips_misses():
    from codenames.spymasters.stop_listener import points_at

    # Pick 1: a neutral word for sure; pick 2: own word 0; pick 3: own word 1; then STOP.
    S = np.full((5, 2 + len(ROLES) + 1), -np.inf)
    S[0, 2] = S[1, 0] = S[2, 1] = 0.0
    S[3:, -1] = 0.0
    assert points_at(S, 2) == pytest.approx(2.0)
    # A coin flip between STOP and own word 0 at pick 1, then STOP: about a half.
    S = np.full((3, 2 + len(ROLES) + 1), -np.inf)
    S[0, [0, -1]] = 0.0
    S[1:, -1] = 0.0
    assert points_at(S, 2) == pytest.approx(0.5, abs=0.03)
