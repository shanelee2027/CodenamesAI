"""The Gaussian spymaster scored by the probability of winning (thesis Section 4).

`expected_words` picks the clue and number with the highest expected reward
under the Gaussian guesser model. This model keeps that guesser model and
changes only what it maximizes: the probability of winning after the turn,

    W(c, k) = sum over endings P(ending | c, k) * (win probability after it),

where the endings are "found j own words, then a neutral / opponent word /
the assassin" and "found all k". The ending probabilities come from the same
integral (`gain_and_penalty`), called with indicator costs: 1 on the opponent
words, then 1 on the assassin. The win probability after an ending comes from
V(a, b), the probability that the side to move wins with a own words left
against b, read from a table (`value_path`). codenames/win_value.py
combines the two; it is the same code win_prob_listener uses with its own
turn model.

V depends on how both sides play, so it is estimated by policy iteration:
self-play of this model against itself with a simulated guesser that follows
the Gaussian model exactly (configs/guesser_gaussian.json), V re-estimated
from those games (scripts/data/build_win_value.py), repeated until V stops
changing. Round 0's V comes from `expected_words` playing itself.
scripts/pipeline/gaussian_policy_iteration.py runs the loop;
docs/versions/gaussian_win_value.md has the results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codenames.board import Role, clue_number_cap
from codenames.spymasters.expected_words import ExpectedWordsSpymaster, gain_and_penalty
from codenames.win_value import WinValue, win_probability


class GaussianWinValueSpymaster(ExpectedWordsSpymaster):
    def __init__(self, *args, value_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_path = Path(value_path)
        self.value = WinValue(self.value_path)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        return [Path(params["value_path"])]

    def _score_matrix(self, a: np.ndarray, b: np.ndarray, costs: np.ndarray,
                      non_own_roles: list[Role], n_opponent: int) -> np.ndarray:
        """W(c, k) for every candidate and k. `costs` (the reward's role costs)
        is not used: the ending probabilities are read off the integral with
        indicator costs instead."""
        def gp(indicator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return gain_and_penalty(a, b, indicator.astype(np.float32), self.sigma, max_number=self.max_number)

        n_own = a.shape[1]
        k_max = clue_number_cap(n_own, self.max_number)
        return win_probability(gp, non_own_roles, k_max, n_own, n_opponent, self.value).astype(np.float32)
