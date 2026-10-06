"""stop_listener valued in net words instead of P(win)
(docs/versions/stop_net_words_listener.md).

Everything but the objective is stop_listener's: the shortlist from its
frozen search, the STOP listener's turn, and the number set to what the clue
points at. Each clue is then valued at its number by the expected net words
of the turn: +1 per own word, minus the role costs for the word that ends it
(`neutral_cost`, `opponent_cost`, `assassin_cost`; by default the
scoreboard's 0.2, 1 and 10), and 0 for STOP. That is the incumbent's reward,
under a guesser that can stop.
"""

from __future__ import annotations

import numpy as np

from codenames.game import Role
from codenames.spymasters.stop_listener import StopListenerSpymaster


def turn_net_words(S: np.ndarray, n_own: int, costs: np.ndarray, K: int,
                   promise_cost: float = 0.0) -> np.ndarray:
    """Expected net words for numbers 1..K. S is (K, n_words + 1) as in
    stop_listener.turn_values; `costs` holds the bad words' costs, in their
    order after the own words. A STOP after j picks on a clue for k costs
    `promise_cost` * (k - j), the promised words not attempted
    (overpromise_listener). Exact, over the set of own words found."""
    n = S.shape[1] - 1
    lam = np.exp(S - S.max(axis=1, keepdims=True))
    out = np.zeros(K)
    for k in range(1, K + 1):
        memo: dict = {}

        def value(found: frozenset) -> float:
            if found in memo:
                return memo[found]
            j, got = len(found), float(len(found))
            if got == n_own or j == k:
                return got
            own_left = [i for i in range(n_own) if i not in found]
            w_own, w_bad, w_stop = lam[j, own_left], lam[j, n_own:n], lam[j, n]
            z = w_own.sum() + w_bad.sum() + w_stop
            v = (w_bad @ (got - costs) + w_stop * (got - promise_cost * (k - j))) / z
            for i, w in zip(own_left, w_own):
                v += w / z * value(found | {i})
            memo[found] = v
            return v

        out[k - 1] = value(frozenset())
    return out


class StopNetWordsListenerSpymaster(StopListenerSpymaster):
    value_kind = "net_words"

    def turn_value(self, S: np.ndarray, n_own: int, roles: list[Role], k: int) -> float:
        """Expected net words of a turn with number k."""
        costs = np.array([self.costs[r] for r in roles], dtype=np.float64)
        return float(turn_net_words(S, n_own, costs, k)[-1])
