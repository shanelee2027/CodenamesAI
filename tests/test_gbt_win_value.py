"""codenames/spymasters/gbt_win_value.py (the GBT spymaster scored by W(c, k)) and the simulated GBT
guesser it is trained against (codenames/guessers/listener_sample.py with pick temperatures)."""

from __future__ import annotations

import numpy as np

from codenames.board import Role
from codenames.guessers.listener_sample import ListenerSampleGuesser
from codenames.spymasters.gbt_win_value import GbtWinValueSpymaster
from codenames.win_value import WinValue

TEMPS = (1.0, 2.0, 3.0)


def make_spymaster(tmp_path) -> GbtWinValueSpymaster:
    """Only the reward model and V are needed, so skip loading the listener."""
    a = np.arange(10)[:, None]
    b = np.arange(10)[None, :]
    V = 1.0 / (1.0 + np.exp(0.6 * (a - b) - 0.3))
    V[0, :], V[:, 0] = 1.0, 0.0
    np.savez(tmp_path / "V.npz", V=V)
    sm = object.__new__(GbtWinValueSpymaster)
    sm.pick_temperatures = TEMPS
    sm.value = WinValue(tmp_path / "V.npz")
    return sm


def test_k1_matches_the_formula_by_hand(tmp_path):
    """W at k = 1 = P(own first) W(stop after 1) + P(neutral first) W(neutral) + P(opponent first) W(opponent),
    pick 1 at temperature TEMPS[0]."""
    sm = make_spymaster(tmp_path)
    s_own = np.array([[2.0, 0.5]])
    s_bad = np.array([[1.0, 0.3, 0.0]])
    roles = [Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN]
    W, unit = sm._clue_values(s_own, s_bad, roles, np.zeros(3), 2, None)
    e = np.exp(np.concatenate([s_own[0], s_bad[0]]) / TEMPS[0])
    p = e / e.sum()
    w = sm.value.after(2, 1)
    expected = (p[0] + p[1]) * w["stop"][1] + p[2] * w["neutral"][0] + p[3] * w["opponent"][0]
    assert W.shape == (1, 2)
    assert np.isclose(W[0, 0], expected)
    assert unit > 0


def test_scores_are_probabilities_and_a_sure_clear_wins(tmp_path):
    sm = make_spymaster(tmp_path)
    s_own = np.array([[30.0, 30.0], [1.0, 0.0]])
    s_bad = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    W, _ = sm._clue_values(s_own, s_bad, [Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN], np.zeros(3), 2, None)
    assert np.all((W >= 0) & (W <= 1))
    assert np.isclose(W[0, 1], 1.0)          # both own words found for sure: the board is cleared


def test_guesser_samples_each_pick_at_its_temperature(monkeypatch):
    """Pick 1 from softmax(s / TEMPS[0]); pick 2, given pick 1, from softmax(s / TEMPS[1]) over the rest."""
    g = object.__new__(ListenerSampleGuesser)
    g.seq, g.seed, g._vw, g.pick_temperatures = None, 0, None, TEMPS
    s = np.array([2.0, 1.0, 0.0])
    monkeypatch.setattr(g, "scores", lambda *a, **k: s)
    words = ["a", "b", "c"]
    n = 4000
    firsts, seconds_after_a = [], []
    for i in range(n):
        order = g.rank_candidates(f"clue{i}", words, None, number=2)
        firsts.append(order[0])
        if order[0] == "a":
            seconds_after_a.append(order[1])
    p1 = np.exp(s / TEMPS[0]) / np.exp(s / TEMPS[0]).sum()
    f = np.mean(np.array(firsts) == "a")
    assert abs(f - p1[0]) < 4 * np.sqrt(p1[0] * (1 - p1[0]) / n)
    p2 = np.exp(s[1] / TEMPS[1]) / (np.exp(s[1] / TEMPS[1]) + np.exp(s[2] / TEMPS[1]))
    m = len(seconds_after_a)
    f2 = np.mean(np.array(seconds_after_a) == "b")
    assert abs(f2 - p2) < 4 * np.sqrt(p2 * (1 - p2) / m)
