"""The estimator has to recover a sigma it was given before any number it
produces from real data can be believed. Every test here plants a known
sigma, simulates the listener the model assumes, and checks the fit finds it.
"""

from __future__ import annotations

import numpy as np
import pytest

from codenames.listener_fit import (
    Observation,
    calibration,
    fit_sigma,
    likelihood_interval,
    log_likelihood,
    profile,
)


def _simulate(n_positions: int, n_words: int, sigma: float, seed: int = 0) -> list[Observation]:
    """Draw boards of z-scores, rank them the way the model says a listener
    with this sigma would, and hand back the result in ranked order."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_positions):
        z = rng.normal(0.0, 1.0, size=n_words)
        perceived = z + rng.normal(0.0, sigma, size=n_words)
        order = np.argsort(-perceived)
        out.append(Observation(clue="x", z=z[order], number=2))
    return out


class TestRecovery:
    @pytest.mark.parametrize("true_sigma", [0.5, 1.0, 2.0, 4.0])
    def test_recovers_a_planted_sigma(self, true_sigma):
        obs = _simulate(1200, 16, true_sigma, seed=int(true_sigma * 100))
        est, _ = fit_sigma(obs)
        assert est == pytest.approx(true_sigma, rel=0.15)

    def test_more_data_tightens_the_estimate(self):
        """RMSE over several seeds, not the error at one: a single draw can
        make the smaller sample look better by luck, which is exactly what a
        one-seed version of this test asserts is impossible."""
        rmse = []
        for n in (100, 1600):
            errs = [fit_sigma(_simulate(n, 16, 2.0, seed=s))[0] - 2.0 for s in range(6)]
            rmse.append(float(np.sqrt(np.mean(np.square(errs)))))
        assert rmse[1] < rmse[0], rmse

    def test_unbiased_across_board_sizes(self):
        """Boards shrink as a game proceeds, so the estimate must not depend
        on how many words were on the board when the listener answered."""
        for n_words in (6, 12, 25):
            est, _ = fit_sigma(_simulate(2000, n_words, 2.0, seed=n_words))
            assert est == pytest.approx(2.0, rel=0.15), n_words

    def test_likelihood_interval_brackets_the_estimate(self):
        obs = _simulate(1500, 16, 2.0, seed=3)
        est, _ = fit_sigma(obs)
        lo, hi = likelihood_interval(obs, est)
        assert lo < est < hi
        assert lo < 2.0 < hi


class TestLikelihoodShape:
    def test_peaks_at_the_truth(self):
        obs = _simulate(800, 12, 2.0, seed=11)
        grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 9.0])
        ll = profile(obs, grid)
        assert grid[int(np.argmax(ll))] == 2.0

    def test_zero_and_negative_sigma_are_impossible(self):
        obs = _simulate(20, 8, 1.0)
        z = np.stack([o.z for o in obs])
        mask = np.ones_like(z, dtype=bool)
        assert log_likelihood(z, mask, 0.0) == -np.inf
        assert log_likelihood(z, mask, -1.0) == -np.inf

    def test_perfectly_ordered_data_drives_sigma_down(self):
        """A listener that always follows z-order exactly is the sigma -> 0
        limit; the fit must not return a comfortable mid-range value."""
        rng = np.random.default_rng(0)
        obs = [Observation(clue="x", z=np.sort(rng.normal(size=10))[::-1].copy(), number=1)
               for _ in range(300)]
        est, _ = fit_sigma(obs)
        assert est < 0.5

    def test_shuffled_data_drives_sigma_up(self):
        """Rankings unrelated to z are the sigma -> inf limit."""
        rng = np.random.default_rng(1)
        obs = [Observation(clue="x", z=rng.normal(size=12), number=1) for _ in range(600)]
        est, _ = fit_sigma(obs, hi=60.0)
        assert est > 8.0


class TestCalibration:
    def test_well_specified_data_calibrates(self):
        """The diagnostic must pass when the model is right, or it cannot be
        evidence of anything when it fails."""
        obs = _simulate(2000, 16, 2.0, seed=5)
        c = calibration(obs, 2.0)
        assert c["simulated_mean_zrank"] == pytest.approx(c["observed_mean_zrank"], rel=0.12)
        assert c["simulated_top1_rate"] == pytest.approx(c["observed_top1_rate"], rel=0.15)

    def test_detects_a_listener_the_model_cannot_represent(self):
        """A listener picking a word numberbatch ranks LAST is not noisy, it
        is using knowledge the embedding lacks. No sigma reproduces that, and
        the diagnostic has to say so -- this is the failure mode the whole
        caution in the module docstring is about."""
        rng = np.random.default_rng(2)
        obs = []
        for _ in range(500):
            z = rng.normal(size=14)
            order = np.argsort(z)  # ascending: listener picks the WORST z first
            obs.append(Observation(clue="x", z=z[order], number=1))
        est, _ = fit_sigma(obs, hi=60.0)
        c = calibration(obs, est)
        assert c["observed_mean_zrank"] > c["simulated_mean_zrank"]
