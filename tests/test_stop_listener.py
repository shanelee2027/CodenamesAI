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


def test_net_words_counts_own_words_and_charges_the_ending():
    from codenames.spymasters.stop_net_words_listener import turn_net_words

    costs = np.array([0.2, 0.2, 1.0, 1.0, 10.0])          # ROLES' costs
    # Pick 1: own word 0 for sure. Pick 2: a third each own word 1, the assassin, STOP.
    S = np.full((2, 2 + len(ROLES) + 1), -np.inf)
    S[0, 0] = 0.0
    S[1, [1, 6, 7]] = 0.0
    v = turn_net_words(S, 2, costs, 2)
    assert v[0] == pytest.approx(1.0)
    assert v[1] == pytest.approx((2.0 + (1.0 - 10.0) + 1.0) / 3)


def test_promise_cost_charges_unattempted_words_at_stop_only():
    from codenames.spymasters.stop_net_words_listener import turn_net_words

    costs = np.array([0.2, 0.2, 1.0, 1.0, 10.0])
    # Pick 1: own word 0 for sure. Pick 2: half STOP, half the assassin. Pick 3: STOP.
    S = np.full((3, 3 + len(ROLES) + 1), -np.inf)
    S[0, 0] = 0.0
    S[1, [7, 8]] = 0.0
    S[2, -1] = 0.0
    v = turn_net_words(S, 3, costs, 3, promise_cost=0.5)
    assert v[0] == pytest.approx(1.0)                                   # the number is reached
    assert v[1] == pytest.approx(0.5 * (1 - 0.5 * 1) + 0.5 * (1 - 10))  # STOP short by 1; the assassin is not charged
    assert v[2] == pytest.approx(0.5 * (1 - 0.5 * 2) + 0.5 * (1 - 10))  # STOP short by 2


def test_win_prob_promise_with_no_charge_is_stop_listeners_value():
    from codenames.spymasters.win_prob_overpromise_listener import turn_values_promise

    rng = np.random.default_rng(3)
    n_own, K = 3, 4
    S = rng.normal(0, 1.5, (K, n_own + len(ROLES) + 1))
    assert turn_values_promise(S, n_own, ROLES, _Value.V, K, 0.0) == pytest.approx(
        turn_values(S, n_own, ROLES, _Value.V, K))


def test_win_prob_promise_charge_lowers_the_value_more_for_larger_numbers():
    from codenames.spymasters.win_prob_overpromise_listener import turn_values_promise

    rng = np.random.default_rng(4)
    n_own, K = 4, 4
    S = rng.normal(0, 1.5, (K, n_own + len(ROLES) + 1))
    free = turn_values_promise(S, n_own, ROLES, _Value.V, K, 0.0)
    charged = turn_values_promise(S, n_own, ROLES, _Value.V, K, 0.05)
    gap = free - charged
    assert np.all(gap >= 0)
    assert np.all(np.diff(gap) > 0)
    # k = 1: the charge is paid only when pick 1 is STOP, once.
    z = np.exp(S[0]).sum()
    assert gap[0] == pytest.approx(0.05 * np.exp(S[0, -1]) / z)


def test_win_prob_promise_builds_from_the_registry_with_the_strict_stop_model():
    from codenames.spymasters.registry import spymaster_spec
    from codenames.spymasters.win_prob_overpromise_listener import WinProbOverpromiseListenerSpymaster

    cls, kw = spymaster_spec("win_prob_overpromise_listener")
    assert cls is WinProbOverpromiseListenerSpymaster
    assert kw["promise_cost"] == 0.5
    files = {p.name for p in cls.model_files(kw)}
    assert "listener_gbt_stop_guess.txt" in files and "listener_gbt_stop.txt" not in files


def test_margin_value_promise_reads_the_margin_table():
    from codenames.spymasters.margin_value_overpromise_listener import MarginValueOverpromiseListenerSpymaster
    from codenames.spymasters.registry import spymaster_spec

    cls, kw = spymaster_spec("margin_value_overpromise_listener")
    assert cls is MarginValueOverpromiseListenerSpymaster
    assert kw["promise_cost"] == 0.5
    files = {p.name for p in cls.model_files(kw)}
    assert "win_value_margin.npz" in files and "win_value.npz" not in files
    assert "listener_gbt_stop_guess.txt" in files


def test_floored_charge_never_falls_below_the_floor():
    """The charge is promise_cost * max(one_word, floor): unchanged where one
    word is worth more than the floor, the floor where it is worth less."""
    from types import SimpleNamespace

    from codenames.spymasters.floored_overpromise_listener import FlooredOverpromiseListenerSpymaster
    from codenames.spymasters.registry import spymaster_spec

    cls, kw = spymaster_spec("floored_overpromise_listener")
    assert cls is FlooredOverpromiseListenerSpymaster
    assert "win_value_margin.npz" in {p.name for p in cls.model_files(kw)}

    sm = object.__new__(FlooredOverpromiseListenerSpymaster)
    worth = {(5, 2): 0.006, (5, 4): 0.116}
    sm.value = SimpleNamespace(one_word=lambda a, b: worth[(a, b)])
    sm.promise_cost, sm.charge_floor = 1.0, 0.05
    roles = lambda b: [Role.OPPONENT] * b + [Role.NEUTRAL]
    assert sm._charge(5, roles(2)) == pytest.approx(0.05)
    assert sm._charge(5, roles(4)) == pytest.approx(0.116)
