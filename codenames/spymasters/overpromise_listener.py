"""stop_net_words_listener with a cost for promising more than the guesser
attempts (docs/versions/overpromise_listener.md).

**Why.** A guesser that can stop makes a large number nearly free: if the
clue runs out, it stops, and the turn loses nothing. The incumbent's number
was held down by the forced guesses after the clue ran out; with STOP that
brake is gone, which is why the stop listeners announce more than the
incumbent (3.77 against 2.93 on the same 30 opening boards), not less. For a
human, a guesser who stops short is not free: it means the clue promised
words it did not deliver.

**The cost.** When the guesser stops after j picks on a clue for k, the turn
is charged `promise_cost` (lambda) for each of the k - j promised words it
did not attempt. A wrong word ends the turn with its own role cost, as
before, and is not charged again: the guesser did attempt it. Announcing one
more word then costs lambda times the chance the guesser stops before
reaching it, and gains that word's expected net words.

**The number is the objective's again:** the k with the best expected net
words minus the cost, in [1, cap], instead of stop_listener's count of what
the clue points at. The penalty is what makes that choice meaningful.
"""

from __future__ import annotations

import numpy as np

from codenames.game import Role
from codenames.spymasters.stop_net_words_listener import StopNetWordsListenerSpymaster, turn_net_words


class OverpromiseListenerSpymaster(StopNetWordsListenerSpymaster):
    def __init__(self, *args, promise_cost: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.promise_cost = float(promise_cost)

    def picks_needed(self, cap: int) -> int:
        return cap

    def _values(self, S: np.ndarray, n_own: int, roles: list[Role], K: int) -> np.ndarray:
        costs = np.array([self.costs[r] for r in roles], dtype=np.float64)
        return turn_net_words(S, n_own, costs, K, self.promise_cost)

    def turn_value(self, S: np.ndarray, n_own: int, roles: list[Role], k: int) -> float:
        """Expected net words of a turn with number k, minus the cost of unattempted promises."""
        return float(self._values(S, n_own, roles, k)[-1])

    def choose(self, S: np.ndarray, n_own: int, roles: list[Role], cap: int) -> tuple[int, float]:
        v = self._values(S[:cap], n_own, roles, cap)
        return int(v.argmax()) + 1, float(v.max())
