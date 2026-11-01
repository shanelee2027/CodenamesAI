"""leftover_plus_one_listener: the clean endings of a turn get one more guess
from the previous clue, and only when the rule says there is a leftover."""

import numpy as np
import pytest

from codenames.board import Board
from codenames.game import Role
from codenames.spymasters.base import PastTurn
from codenames.spymasters.leftover_plus_one_listener import (LeftoverPlusOneListenerSpymaster, still_unfound,
                                                             turn_values_leftover)
from codenames.spymasters.win_prob_overpromise_listener import turn_values_promise

V = np.random.default_rng(1).uniform(0.2, 0.8, (10, 10))
ROLES = [Role.NEUTRAL, Role.NEUTRAL, Role.OPPONENT, Role.OPPONENT, Role.ASSASSIN]


def _S(seed=0, n_own=3, K=3):
    return np.random.default_rng(seed).normal(0, 1.5, (K, n_own + len(ROLES) + 1))


def test_no_plus_one_is_the_promise_recursion():
    S, n_own = _S(), 3
    L = np.random.default_rng(2).dirichlet(np.ones(n_own + len(ROLES) + 1))
    want = turn_values_promise(S, n_own, ROLES, V, 3, 0.02)
    assert turn_values_leftover(S, n_own, ROLES, V, 3, 0.02, L, 0.0) == pytest.approx(want)
    only_stop = np.zeros_like(L)
    only_stop[-1] = 1.0
    assert turn_values_leftover(S, n_own, ROLES, V, 3, 0.02, only_stop, 1.0) == pytest.approx(want)


def test_a_sure_own_leftover_is_one_more_word_at_a_clean_ending():
    # Two own words. The clue for 1 finds own word 0 for sure; the +1 then
    # finds own word 1 for sure, which wins.
    S = np.full((1, 2 + len(ROLES) + 1), -np.inf)
    S[0, 0] = 0.0
    L = np.zeros(2 + len(ROLES) + 1)
    L[[0, 1]] = 0.5                       # own word 0 is found by then, so all of it goes to word 1
    assert turn_values_leftover(S, 2, ROLES, V, 1, 0.0, L, 1.0)[0] == pytest.approx(1.0)
    half = turn_values_leftover(S, 2, ROLES, V, 1, 0.0, L, 0.5)[0]
    assert half == pytest.approx(0.5 * 1.0 + 0.5 * (1 - V[2, 1]))


def test_the_plus_one_can_hit_the_assassin_and_a_wrong_guess_gets_no_plus_one():
    # Pick 1: the own word or a neutral, half each. The +1 is the assassin.
    S = np.full((1, 2 + len(ROLES) + 1), -np.inf)
    S[0, [0, 2]] = 0.0
    L = np.zeros(2 + len(ROLES) + 1)
    L[6] = 1.0
    got = turn_values_leftover(S, 2, ROLES, V, 1, 0.0, L, 1.0)[0]
    assert got == pytest.approx(0.5 * 0.0 + 0.5 * (1 - V[2, 2]))


def test_still_unfound():
    assert still_unfound(1, []) == 1.0
    assert still_unfound(1, [0.3, 0.5]) == pytest.approx(0.7 * 0.5)
    assert still_unfound(2, [0.3, 0.5]) == pytest.approx(1 - 0.3 * 0.5)


class _Fake:
    """Just what `leftover` reads: the board split and flat STOP scores."""
    plus_one_rate = 1.0
    _board = staticmethod(LeftoverPlusOneListenerSpymaster._board)

    def stop_scores(self, clues, words, K, sims):
        return {0: np.zeros((K, len(words) + 1))}


def _board_and_words():
    board = Board.generate(seed=7)
    own = board.words_by_role(Role.OWN)
    bad = board.words_by_role(Role.NEUTRAL)
    opp = board.words_by_role(Role.OPPONENT)
    return board, own, bad, opp


def test_leftover_only_after_a_turn_cut_short_with_words_unfound():
    board, own, bad, opp = _board_and_words()
    f = _Fake()
    get = lambda h: LeftoverPlusOneListenerSpymaster.leftover(f, board, tuple(h), None)   # noqa: E731
    board.revealed = {own[0], bad[0]}
    cut = [PastTurn(True, "CLUE", 2, (own[0], bad[0]))]
    L, q = get(cut)
    assert q == 1.0 and L.shape == (len(board.words) - 2 + 1,)
    assert get([PastTurn(True, "CLUE", 1, (own[0], bad[0]))]) is None        # the number was found
    board.revealed = {own[0], own[1]}
    assert get([PastTurn(True, "CLUE", 3, (own[0], own[1]))]) is None        # stopped, not cut short
    assert get([]) is None
    # The opponent turned over one of our words since: with flat scores it
    # was the clue's with probability 1 / (words then + STOP).
    board.revealed = {own[0], bad[0], opp[0], own[2]}
    h = cut + [PastTurn(False, "OTHER", 2, (opp[0], own[2]))]
    _, q = get(h)
    assert q == pytest.approx(1 - 1 / (len(board.words) - 2 + 1))
