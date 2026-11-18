"""The GBT spymaster scored by the probability of winning (docs/versions/gbt_win_value.md).

`fresh_split_listener` (the thesis's GBT spymaster) picks the clue and number
with the highest expected reward under its model of the guesser: the GBT's
scores, with a temperature per pick. This model keeps that guesser model and
the same search, and changes only what it maximizes, exactly as
`gaussian_win_value` does for the Gaussian spymaster:

    W(c, k) = sum over endings P(ending | c, k) * (win probability after it).

The ending probabilities come from the same exact computation over picked
sets (`tempered_gain_and_penalty`), called with indicator costs: 1 on the
opponent words, then 1 on the assassin. The win probability after an ending
comes from V(a, b), the probability that the side to move wins with a own
words left against b, read from a table (`value_path`).
codenames/win_value.py combines the two; it is the same code
gaussian_win_value and win_prob_listener use.

V depends on how both sides play, so it is estimated by policy iteration:
self-play of this model against itself with a simulated guesser that follows
the GBT exactly (`sim:fresh_split`, codenames/guessers/listener_sample.py),
V re-estimated from those games (scripts/data/build_win_value.py), repeated
until V stops changing. Round 0's V comes from `fresh_split_listener`
playing itself. scripts/pipeline/gbt_policy_iteration.py runs the loop.

The shortlist is the parent's: the Gaussian expected reward picks
`shortlist` clues and W ranks them. With `shortlist` at the whole pool there
is no first stage in effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codenames.board import Role
from codenames.spymasters.fresh_split_listener import FreshSplitListenerSpymaster
from codenames.win_value import WinValue, win_probability


class GbtWinValueSpymaster(FreshSplitListenerSpymaster):
    def __init__(self, *args, value_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_path = Path(value_path)
        self.value = WinValue(self.value_path)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        return [*super().model_files(params), Path(params["value_path"])]

    def _clue_values(self, s_own, s_bad, roles, costs, max_k, s_out, words=None):
        """`(W, one own word in W)`. `costs` (the reward's role costs) is not
        used: the ending probabilities are read off the reward model with
        indicator costs instead."""
        s_own = np.asarray(s_own, dtype=np.float64)
        s_bad = np.asarray(s_bad, dtype=np.float64)

        def gp(indicator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return self._gain_and_penalty(s_own, s_bad, indicator, max_k, s_out, words=words)

        n_own = s_own.shape[1]
        n_opp = sum(r == Role.OPPONENT for r in roles)
        values = win_probability(gp, roles, max_k, n_own, n_opp, self.value)
        return values, self.value.one_word(n_own, n_opp)
