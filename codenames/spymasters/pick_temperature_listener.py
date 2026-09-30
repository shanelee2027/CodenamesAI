"""learned_listener with the listener's confidence set separately for each pick.

**The problem** (docs/log.md, "Is the listener as sure of its later picks as
it should be?"). The incumbent's reward scores a clue once and, after each
pick, renormalises the same scores over the words left (Plackett-Luce,
codenames/pl_reward.py). So its confidence at pick 1 carries into picks 2-4.
Measured on rankings no model was fitted on, pick 1 is calibrated and every
later pick is overconfident, increasingly so. On held-out-word boards the
favourite at pick 3 is predicted 53% and picked 34%. A clue whose later
words are weak is therefore overvalued, which is the "95 / 3 / 1 / 1 for 4"
pattern.

**The change.** At pick j the guesser picks from the words left with
probabilities softmax(s / tau_j). tau_j > 1 flattens pick j. The temperatures
are fitted per pick by maximum likelihood on rankings the listener never saw
(scripts/tools/listener_step_calibration.py). Nothing else differs from
learned_listener: the same listener, the same shortlist, the same costs, the
same search.

**Why the reward is recomputed, not patched.** pl_reward's closed form
(exponential clocks) holds only when the scores are fixed across picks, and a
temperature that changes with the pick breaks that. Instead the expectation is
computed exactly over which of our words have been picked so far. A wrong pick
ends the turn, so only our own words branch. The probability of having picked
exactly the set A after |A| picks is a sum over its last member:

    P(A) = sum over o in A of P(A - {o}) * p_|A|(o | words left after A - {o})

With 9 own words that is 2^9 = 512 sets, each updated from at most 9
predecessors, so it is exact for any announced number, uncapped included. The
turn's value for announcing k is

    gain(k)    = sum over j <= k of P(all of the first j picks are ours)
    penalty(k) = sum over j <= k of sum over |A| = j-1 of
                 P(A) * sum over non-team b of p_j(b | left) * cost(b)

which is what pl_reward computes, and at tau = 1 the two agree
(tests/test_pick_temperature_listener.py). The outside option, when on, is a
non-team word with cost 0, exactly as in pl_reward.
"""

from __future__ import annotations

import numpy as np

from codenames.spymasters.learned_listener import LearnedListenerSpymaster

# Fitted on the clean-holdout boards (seed 1,040,000 on) for cache/listener_gbt.txt;
# docs/versions/pick_temperature_listener.md has the fit. Pick 4's value is
# used for every pick after it.
PICK_TEMPERATURES = (1.0, 1.1, 1.3, 1.5)


def tempered_gain_and_penalty(s_own: np.ndarray, s_bad: np.ndarray, costs: np.ndarray, max_k: int,
                              temperatures: tuple[float, ...],
                              s_out: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(gain, penalty)`, each `(n_cand, max_k)`, column m being k = m + 1, as
    pl_reward.gain_and_penalty returns, with pick j scored at temperature
    `temperatures[min(j, len) - 1]`."""
    s_own = np.asarray(s_own, dtype=np.float64)
    s_bad = np.asarray(s_bad, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if s_out is not None:
        s_bad = np.concatenate([s_bad, np.asarray(s_out, dtype=np.float64).reshape(-1, 1)], axis=1)
        costs = np.concatenate([costs, [0.0]])
    C, n_own = s_own.shape
    max_k = min(max_k, n_own)
    shift = np.maximum(s_own.max(1, keepdims=True), s_bad.max(1, keepdims=True) if s_bad.size else -np.inf)

    n_sub = 1 << n_own
    members = (np.arange(n_sub)[:, None] >> np.arange(n_own)[None, :]) & 1      # (n_sub, n_own)
    size = members.sum(1)
    P = np.zeros((C, n_sub))
    P[:, 0] = 1.0
    reach = np.zeros((C, max_k))
    pen = np.zeros((C, max_k))
    for j in range(1, max_k + 1):
        tau = temperatures[min(j, len(temperatures)) - 1]
        e_own = np.exp((s_own - shift) / tau)                                    # (C, n_own)
        e_bad = np.exp((s_bad - shift) / tau)                                    # (C, n_bad)
        bad_mass, bad_cost = e_bad.sum(1), e_bad @ costs                         # (C,)
        S = np.flatnonzero(size == j - 1)
        Z = bad_mass[:, None] + e_own.sum(1, keepdims=True) - e_own @ members[S].T  # (C, |S|)
        Z = np.maximum(Z, 1e-300)
        w = P[:, S] / Z
        pen[:, j - 1] = (w * bad_cost[:, None]).sum(1)
        for o in range(n_own):
            free = members[S, o] == 0
            P[:, S[free] | (1 << o)] += w[:, free] * e_own[:, o:o + 1]
        reach[:, j - 1] = P[:, size == j].sum(1)
    return np.cumsum(reach, 1).astype(np.float32), np.cumsum(pen, 1).astype(np.float32)


class PickTemperatureListenerSpymaster(LearnedListenerSpymaster):
    def __init__(self, *args, pick_temperatures: tuple[float, ...] | list[float] = PICK_TEMPERATURES, **kwargs):
        super().__init__(*args, **kwargs)
        self.pick_temperatures = tuple(float(t) for t in pick_temperatures)

    def _gain_and_penalty(self, s_own, s_bad, costs, max_k, s_out, words=None):
        return tempered_gain_and_penalty(s_own, s_bad, costs, max_k, self.pick_temperatures, s_out)
