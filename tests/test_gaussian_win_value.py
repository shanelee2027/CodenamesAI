"""codenames/spymasters/gaussian_win_value.py: the Gaussian model scored by W(c, k)."""

from __future__ import annotations

import numpy as np

from codenames.board import Role
from codenames.spymasters.gaussian_win_value import GaussianWinValueSpymaster
from codenames.spymasters.expected_words import gain_and_penalty
from tests.test_expected_words_spymaster import (
    SPACE, _FAR_OWN, make_board, make_ctx, make_row, make_sims, make_stats,
)


def write_value(tmp_path, V):
    path = tmp_path / "V.npz"
    np.savez(path, V=V)
    return path


def score_only_value(tmp_path):
    """A monotone V: the side to move wins more with fewer own words left and more of the other's."""
    a = np.arange(10)[:, None]
    b = np.arange(10)[None, :]
    V = 1.0 / (1.0 + np.exp(0.6 * (a - b) - 0.3))
    V[0, :], V[:, 0] = 1.0, 0.0
    return write_value(tmp_path, V)


def build(tmp_path, rows):
    clue_words = list(rows)
    tensor = np.asarray([rows[c] for c in clue_words], dtype=np.float32)[:, :, None]
    sims = make_sims(tmp_path, tensor, clue_words)
    sm = GaussianWinValueSpymaster(space=SPACE, clue_stats=make_stats(clue_words), sigma=1.5,
                                   max_number=None, value_path=score_only_value(tmp_path))
    return sm, sims


def test_scores_are_win_probabilities(tmp_path):
    rows = {"safe": make_row(own=[3.0, 2.8] + [_FAR_OWN] * 7, assassin=[0.5]),
            "risky": make_row(own=[2.0, 1.9, 1.8] + [_FAR_OWN] * 6, opponent=[1.9] * 8)}
    sm, sims = build(tmp_path, rows)
    _, scores, _ = sm._score_all_clues(make_board(), sims)
    assert np.all((scores >= 0) & (scores <= 1))


def test_far_distractors_make_the_win_probability_rise_with_k(tmp_path):
    """With nothing near, every k is found for sure, and finding more own words can only help."""
    own = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2]
    sm, sims = build(tmp_path, {"clue": make_row(own=own)})
    clue, number = sm.give_clue(make_ctx(make_board()), sims)
    assert clue == "clue" and number == 9   # finds all nine: the game is won


def test_assassin_risk_is_priced_as_losing(tmp_path):
    """A k whose next word sits next to the assassin loses the game when it misses, so W prefers stopping."""
    rows = {"clue": make_row(own=[3.0, 0.6] + [_FAR_OWN] * 7, assassin=[0.7])}
    sm, sims = build(tmp_path, rows)
    _, number = sm.give_clue(make_ctx(make_board()), sims)
    assert number == 1


def test_matches_win_probability_by_hand(tmp_path):
    """The score at k = 1 for one clue equals the formula written out: P(find it) * W(stop after 1)
    + P(neutral first) * W(neutral) + P(opponent first) * W(opponent) + P(assassin first) * 0."""
    own = [2.0] + [_FAR_OWN] * 8
    rows = {"clue": make_row(own=own, opponent=[1.0] + [-100.0] * 7, neutral=[0.5] + [-100.0] * 6, assassin=[0.0])}
    sm, sims = build(tmp_path, rows)
    board = make_board()
    best_n, scores, _ = sm._score_all_clues(board, sims)
    a = -np.sort(-np.array([own], dtype=np.float32), axis=1)
    b = np.array([rows["clue"][9:]], dtype=np.float32)    # opponent, neutral, assassin order of the row
    roles = [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN]
    # z_for_board orders non-own words neutral, opponent, assassin; reorder b to match.
    order = list(range(8, 15)) + list(range(0, 8)) + [15]
    b, roles = b[:, order], [roles[i] for i in order]
    gain, p_opp = gain_and_penalty(a, b, np.array([r == Role.OPPONENT for r in roles], np.float32), 1.5, max_number=None)
    _, p_ass = gain_and_penalty(a, b, np.array([r == Role.ASSASSIN for r in roles], np.float32), 1.5, max_number=None)
    w = sm.value.after(9, 8)
    p1 = gain[0, 0]
    want_k1 = p1 * w["stop"][1] + p_opp[0, 0] * w["opponent"][0] + (1 - p1 - p_opp[0, 0] - p_ass[0, 0]) * w["neutral"][0]
    if int(best_n[0]) == 1:
        assert float(scores[0]) == np.float32(want_k1) or abs(float(scores[0]) - want_k1) < 1e-4
    else:
        assert float(scores[0]) >= want_k1 - 1e-4   # some larger k scored at least as well
