"""The claim in docs/clue-selection-learned.tex is that the exponential race
IS the selection tree, not an approximation of it. That is worth asserting
rather than believing, so the central test here enumerates the tree directly
on boards small enough to brute-force and demands agreement."""

from __future__ import annotations


import numpy as np
import pytest

from codenames.pl_reward import expected_reward, gain_and_penalty


def brute_force(s_own: np.ndarray, s_bad: np.ndarray, costs: np.ndarray, k: int) -> tuple[float, float]:
    """(gain, penalty) by explicit recursion over the selection tree.

    This is the definition the closed form has to match: pick a word with
    probability proportional to exp(score) over whatever is left, and if it is
    ours, recurse on the smaller board with one fewer guess.
    """
    lam_o = np.exp(s_own.astype(np.float64))
    lam_b = np.exp(s_bad.astype(np.float64))

    def rec(own: frozenset[int], bad: frozenset[int], left: int) -> tuple[float, float]:
        if left == 0 or (not own and not bad):
            return 0.0, 0.0
        total = sum(lam_o[i] for i in own) + sum(lam_b[w] for w in bad)
        gain = pen = 0.0
        for i in own:
            pr = lam_o[i] / total
            g, p = rec(own - {i}, bad, left - 1)
            gain += pr * (1.0 + g)
            pen += pr * p
        for w in bad:
            pr = lam_b[w] / total
            pen += pr * float(costs[w])          # turn ends here
        return gain, pen

    return rec(frozenset(range(len(s_own))), frozenset(range(len(s_bad))), k)


class TestMatchesTheSelectionTree:
    @pytest.mark.parametrize("n_own,n_bad,k", [(1, 1, 1), (2, 2, 1), (3, 2, 2),
                                               (3, 3, 3), (4, 3, 2), (2, 4, 4)])
    def test_agrees_with_brute_force(self, n_own, n_bad, k):
        rng = np.random.default_rng(n_own * 100 + n_bad * 10 + k)
        s_own = rng.normal(0.0, 1.5, size=(1, n_own))
        s_bad = rng.normal(0.0, 1.5, size=(1, n_bad))
        costs = rng.uniform(0.2, 10.0, size=n_bad)

        g, p = gain_and_penalty(s_own, s_bad, costs, k, cells=4096)
        bg, bp = brute_force(s_own[0], s_bad[0], costs, k)
        assert g[0, k - 1] == pytest.approx(bg, abs=2e-3), "gain"
        assert p[0, k - 1] == pytest.approx(bp, abs=2e-3), "penalty"

    def test_agrees_across_every_k_in_one_pass(self):
        """Column m must be k = m+1 for every m, not only the last."""
        rng = np.random.default_rng(7)
        s_own = rng.normal(0.0, 1.0, size=(1, 4))
        s_bad = rng.normal(0.0, 1.0, size=(1, 3))
        costs = np.array([0.2, 1.0, 10.0])
        g, p = gain_and_penalty(s_own, s_bad, costs, 4, cells=4096)
        for k in (1, 2, 3, 4):
            bg, bp = brute_force(s_own[0], s_bad[0], costs, k)
            assert g[0, k - 1] == pytest.approx(bg, abs=2e-3)
            assert p[0, k - 1] == pytest.approx(bp, abs=2e-3)

    def test_matches_a_simulation(self):
        """A second, independent check: play the process and count."""
        rng = np.random.default_rng(11)
        s_own = np.array([[1.2, 0.4, -0.3]])
        s_bad = np.array([[0.8, -0.5]])
        costs = np.array([1.0, 10.0])
        k = 2
        lam = np.exp(np.concatenate([s_own[0], s_bad[0]]))
        is_own = np.array([True] * 3 + [False] * 2)

        gains, pens, n_trials = 0.0, 0.0, 200_000
        for _ in range(n_trials):
            # Sampling the race directly, which is also how the identity is
            # stated -- exponential clocks, read in order.
            order = np.argsort(rng.exponential(1.0 / lam))
            got = 0
            for idx in order[:k]:
                if is_own[idx]:
                    got += 1
                else:
                    pens += costs[idx - 3]
                    break
            gains += got
        g, p = gain_and_penalty(s_own, s_bad, costs, k, cells=4096)
        assert g[0, k - 1] == pytest.approx(gains / n_trials, abs=0.02)
        assert p[0, k - 1] == pytest.approx(pens / n_trials, abs=0.05)


class TestProperties:
    def test_first_pick_probability_is_the_softmax(self):
        """With k=1 and zero cost, gain is P(first pick is ours) exactly."""
        s_own = np.array([[2.0, 0.5]])
        s_bad = np.array([[1.0, -1.0]])
        lam = np.exp(np.array([2.0, 0.5, 1.0, -1.0]))
        g, _ = gain_and_penalty(s_own, s_bad, np.zeros(2), 1, cells=4096)
        assert g[0, 0] == pytest.approx(lam[:2].sum() / lam.sum(), abs=2e-3)

    def test_penalty_is_the_softmax_cost_when_k_is_one(self):
        s_own = np.array([[0.3]])
        s_bad = np.array([[0.7, -0.2]])
        costs = np.array([1.0, 10.0])
        lam = np.exp(np.array([0.3, 0.7, -0.2]))
        _, p = gain_and_penalty(s_own, s_bad, costs, 1, cells=4096)
        expect = (lam[1] * costs[0] + lam[2] * costs[1]) / lam.sum()
        assert p[0, 0] == pytest.approx(expect, abs=2e-3)

    def test_gain_is_monotone_in_k_and_capped(self):
        rng = np.random.default_rng(3)
        s_own = rng.normal(size=(5, 6))
        s_bad = rng.normal(size=(5, 8))
        g, _ = gain_and_penalty(s_own, s_bad, np.ones(8), 4)
        assert np.all(np.diff(g, axis=1) >= -1e-6), "more guesses cannot gain less"
        assert np.all(g <= np.arange(1, 5)[None, :] + 1e-6), "gain cannot exceed k"

    def test_scores_are_shift_invariant(self):
        """Only ratios of exp(score) enter, so a constant must change nothing.
        This is what lets the implementation subtract the row max for stability."""
        rng = np.random.default_rng(5)
        s_own = rng.normal(size=(3, 4))
        s_bad = rng.normal(size=(3, 5))
        costs = rng.uniform(0.5, 5.0, size=5)
        a = gain_and_penalty(s_own, s_bad, costs, 3)
        b = gain_and_penalty(s_own + 17.0, s_bad + 17.0, costs, 3)
        np.testing.assert_allclose(a[0], b[0], atol=1e-6)
        np.testing.assert_allclose(a[1], b[1], atol=1e-6)

    def test_survives_extreme_scores(self):
        """A confident row must not overflow or produce NaN."""
        s_own = np.array([[300.0, -300.0]])
        s_bad = np.array([[-300.0, -300.0]])
        g, p = gain_and_penalty(s_own, s_bad, np.array([1.0, 10.0]), 2)
        assert np.all(np.isfinite(g)) and np.all(np.isfinite(p))
        assert g[0, 0] == pytest.approx(1.0, abs=1e-3), "a dominant own word is taken"

    def test_quadrature_is_converged_at_the_default(self):
        rng = np.random.default_rng(13)
        s_own = rng.normal(size=(4, 7))
        s_bad = rng.normal(size=(4, 9))
        costs = rng.uniform(0.2, 10.0, size=9)
        coarse = gain_and_penalty(s_own, s_bad, costs, 4)
        fine = gain_and_penalty(s_own, s_bad, costs, 4, cells=8192)
        assert np.max(np.abs(coarse[0] - fine[0])) < 5e-3, "gain not converged"
        assert np.max(np.abs(coarse[1] - fine[1])) < 5e-3, "penalty not converged"


class TestExpectedReward:
    def test_picks_the_best_k(self):
        rng = np.random.default_rng(17)
        s_own = rng.normal(size=(6, 5))
        s_bad = rng.normal(size=(6, 7))
        costs = rng.uniform(0.2, 10.0, size=7)
        gain, pen = gain_and_penalty(s_own, s_bad, costs, 4)
        best_k, value = expected_reward(s_own, s_bad, costs, 4)
        net = gain - pen
        np.testing.assert_allclose(value, net.max(axis=1), atol=1e-6)
        assert np.all(best_k >= 1) and np.all(best_k <= 4)

    def test_rejects_mismatched_costs(self):
        with pytest.raises(ValueError, match="costs"):
            gain_and_penalty(np.zeros((1, 2)), np.zeros((1, 3)), np.ones(2), 2)


class TestTheOutsideOption:
    """`s_out` prices "the guesser picks something the clue never meant".

    Its whole purpose is to break the shift-invariance the rest of the module
    relies on. Without an anchor the softmax scores a clue on the ordering it
    induces and never on how confidently it induces it, so a clue that barely
    points at its own words and one that points hard at them are worth the same.
    """

    @staticmethod
    def _board(n_own=3, n_bad=4, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.normal(size=(1, n_own)), rng.normal(size=(1, n_bad)),
                rng.uniform(0.2, 2.0, size=n_bad))

    def test_omitting_it_reproduces_the_old_result_exactly(self):
        """Every recorded result predates the outside option, so the default
        path must be bit-identical, not merely close."""
        s_own, s_bad, costs = self._board()
        a = gain_and_penalty(s_own, s_bad, costs, 3)
        b = gain_and_penalty(s_own, s_bad, costs, 3, s_out=None)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_it_is_exactly_a_free_non_team_word(self, ):
        """The semantics are "hitting it ends the turn and scores nothing", so
        passing it via s_out must equal listing it as a bad word costing 0.
        Checked against the brute-force tree, not against the fast path."""
        s_own, s_bad, costs = self._board()
        out = np.array([[0.3]])
        for k in (1, 2, 3):
            g, p = gain_and_penalty(s_own, s_bad, costs, k, s_out=out[:, 0])
            bg, bp = brute_force(s_own[0], np.concatenate([s_bad[0], out[0]]),
                                 np.concatenate([costs, [0.0]]), k)
            assert g[0, k - 1] == pytest.approx(bg, abs=2e-3)
            assert p[0, k - 1] == pytest.approx(bp, abs=2e-3)

    def test_a_negligible_outside_option_changes_nothing(self):
        s_own, s_bad, costs = self._board()
        base = gain_and_penalty(s_own, s_bad, costs, 3)
        tiny = gain_and_penalty(s_own, s_bad, costs, 3, s_out=np.array([-50.0]))
        assert np.allclose(base[0], tiny[0], atol=1e-6)
        assert np.allclose(base[1], tiny[1], atol=1e-6)

    def test_raising_it_monotonically_lowers_the_gain(self):
        """A clue the guesser is more likely to wander off ends its turn
        sooner, so fewer of our words get claimed."""
        s_own, s_bad, costs = self._board()
        gains = [gain_and_penalty(s_own, s_bad, costs, 3,
                                  s_out=np.array([v]))[0][0, 2]
                 for v in (-6.0, -3.0, -1.0, 0.0, 2.0)]
        assert all(a > b for a, b in zip(gains, gains[1:])), gains

    def test_it_drives_a_vague_clue_toward_zero_not_toward_a_penalty(self):
        """The reason cost is 0 rather than a neutral's cost: a clue pointing
        at nothing should be worthless, not catastrophic. Both halves must
        fall, or the model would simply avoid vagueness by fearing it."""
        s_own, s_bad, costs = self._board()
        g0, p0 = gain_and_penalty(s_own, s_bad, costs, 3)
        g1, p1 = gain_and_penalty(s_own, s_bad, costs, 3, s_out=np.array([4.0]))
        assert g1[0, 2] < g0[0, 2]
        assert p1[0, 2] < p0[0, 2], "free endings must lower the expected miss cost"
        assert abs(g1[0, 2] - p1[0, 2]) < abs(g0[0, 2] - p0[0, 2])

    def test_it_breaks_shift_invariance_which_is_the_entire_point(self):
        """Adding a constant to every board score is invisible to the softmax.
        With a fixed anchor it must not be -- that is what makes "how good is
        this clue" answerable at all."""
        s_own, s_bad, costs = self._board()
        out = np.array([0.0])
        a = gain_and_penalty(s_own, s_bad, costs, 3, s_out=out)[0]
        b = gain_and_penalty(s_own + 2.0, s_bad + 2.0, costs, 3, s_out=out)[0]
        assert not np.allclose(a, b, atol=1e-3)
        assert b[0, 2] > a[0, 2], "a clue that points harder should claim more"

    def test_expected_reward_threads_it_through(self):
        """The contract is just that the argument reaches gain_and_penalty."""
        s_own, s_bad, costs = self._board()
        out = np.array([1.0])
        g, p = gain_and_penalty(s_own, s_bad, costs, 3, s_out=out)
        k, v = expected_reward(s_own, s_bad, costs, 3, s_out=out)
        best = np.argmax(g - p, axis=1)
        assert k[0] == best[0] + 1
        assert v[0] == pytest.approx((g - p)[0, best[0]])

    def test_a_good_clue_is_worth_less_once_wandering_is_priced(self):
        s_own = np.array([[3.0, 2.5, 2.0]])          # clue points hard at ours
        s_bad, costs = np.array([[-1.0, -1.5]]), np.array([1.0, 1.0])
        _, v0 = expected_reward(s_own, s_bad, costs, 3)
        _, v1 = expected_reward(s_own, s_bad, costs, 3, s_out=np.array([1.0]))
        assert v1[0] < v0[0]

    def test_a_DANGEROUS_clue_is_worth_MORE_once_wandering_is_priced(self):
        """Counterintuitive, and a direct consequence of costing the outside
        option at zero: it competes with the assassin for the same probability
        mass, so a clue that mostly pointed at danger now mostly points at a
        harmless nothing instead.

        This is the modelling choice's weak flank, and it runs against the
        measurement -- in the probe, turns where a decoy won had 7.6x the base
        assassin rate, so a real guesser wandering off is MORE dangerous, not
        less. The reward is therefore optimistic for exactly the clues it
        should fear most. Recorded here rather than hidden because it is the
        argument for eventually pricing the outside option above zero.
        """
        s_own = np.array([[-1.0]])                   # clue barely points at ours
        s_bad, costs = np.array([[2.0, 1.5]]), np.array([10.0, 10.0])   # assassin-ish
        _, v0 = expected_reward(s_own, s_bad, costs, 3)
        _, v1 = expected_reward(s_own, s_bad, costs, 3, s_out=np.array([3.0]))
        assert v0[0] < 0 and v1[0] > v0[0]
