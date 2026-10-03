"""pick_index_lookahead_listener's turn enumeration: with the same scores at
every pick it is reply_lookahead_listener's frozen one, and with different
scores per pick it uses pick j's scores at pick j."""

import numpy as np
import pytest

from codenames.game import Role
from codenames.spymasters.pick_index_lookahead_listener import (PickIndexLookaheadListenerSpymaster,
                                                                pick_turn_outcomes)
from codenames.spymasters.reply_lookahead_listener import turn_outcomes


def test_same_scores_at_every_pick_is_the_frozen_enumeration():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1.5, 9)
    for k in (1, 2, 3):
        a = turn_outcomes(s, 4, k, prune=0.0)
        b = pick_turn_outcomes(np.tile(s, (k, 1)), 4, k, prune=0.0)
        assert a.keys() == b.keys()
        assert all(abs(a[x] - b[x]) < 1e-12 for x in a)


def test_pick_j_uses_its_own_scores():
    # Own words 0 and 1, bad word 2. Pick 1 by S[0], pick 2 by S[1] among the rest.
    S = np.log(np.array([[2.0, 1.0, 1.0],
                         [1.0, 1.0, 3.0]]))
    out = pick_turn_outcomes(S, 2, 2, prune=0.0)
    p_first = {0: 0.5, 1: 0.25, 2: 0.25}
    both = p_first[0] * 1 / (1 + 3) + p_first[1] * 1 / (1 + 3)       # the other own word, at pick 2
    assert out[(frozenset({0, 1}), None)] == pytest.approx(both)
    assert out[(frozenset({0}), 2)] == pytest.approx(p_first[0] * 3 / 4)
    assert out[(frozenset({1}), 2)] == pytest.approx(p_first[1] * 3 / 4)
    assert out[(frozenset(), 2)] == pytest.approx(p_first[2])
    assert sum(out.values()) == pytest.approx(1.0)


class _Value:
    V = np.full((10, 10), 0.4)                      # the mover's P(win) at every score


def test_turn_value_counts_wins_losses_and_v():
    sm = PickIndexLookaheadListenerSpymaster.__new__(PickIndexLookaheadListenerSpymaster)
    sm.value = _Value()
    roles = [Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN]           # words 2, 3, 4; two own words 0, 1
    outcomes = {(frozenset({0, 1}), None): 0.2,                  # all own words found: a win
                (frozenset({0}), 4): 0.3,                         # the assassin: a loss
                (frozenset({0}), 2): 0.4,                         # neutral: 1 - V
                (frozenset(), 3): 0.1}                            # the opponent's last word: a loss
    assert sm.turn_value(outcomes, 2, roles) == pytest.approx(0.2 + 0.4 * 0.6)
    # Pruned mass is not counted as a loss.
    half = {k: v / 2 for k, v in outcomes.items()}
    assert sm.turn_value(half, 2, roles) == pytest.approx(0.2 + 0.4 * 0.6)


def test_builds_from_the_registry_with_its_own_files():
    from codenames.spymasters.registry import spymaster_spec

    cls, kw = spymaster_spec("pick_index_lookahead_listener")
    files = {p.name for p in cls.model_files(kw)}
    assert {"listener_gbt_assoc_features.txt", "listener_gbt_pick_indexassoc_depth9.txt",
            "reply_offset_pick_index.npz", "listener_gbt.txt"} <= files
    assert "reply_offset.npz" not in files
    try:
        sm = cls(**kw)
    except FileNotFoundError:
        pytest.skip("needs the cached boosters and offset table")
    assert callable(sm.top_clues) and callable(sm.give_clue)
