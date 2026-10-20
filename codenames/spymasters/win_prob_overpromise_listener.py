"""stop_listener's P(win) objective with overpromise_listener's cost for
promised words the guesser does not attempt
(docs/versions/win_prob_overpromise_listener.md).

**Why.** On the comfort arena's scoreboard (docs/log.md, 2026-10-18) the
frontier ran from win_prob (most wins, least comfortable) to the strict
overpromise models (most comfortable). The two halves are separate models:
win_prob values a turn by the chance of winning after it but never thinks
about promises; overpromise charges for broken promises but values the turn
in net words. This puts the charge on the win-probability objective.

**The value.** The turn under the STOP listener as in stop_listener
(`turn_values`: an exact recursion over the own words found, each pick a
softmax over the words left and STOP), with P(win) after the turn at every
ending. A STOP after j picks on a clue for k is charged
promise_cost * (k - j) * w, where w = `WinValue.one_word(a, b)` is what one
more own word is worth now, in win probability, at the turn's starting score
(a own and b opponent words left). So promise_cost is in words, as in
overpromise_listener -- 0.5 means each promised word not attempted costs half
an own word -- and converted to win probability at the current score: a
broken promise costs more where a word matters more. A wrong word is charged
only through the state it leaves (the guesser attempted it), and a turn that
finds our last word is a win with no charge.

**The number** is the objective's: the k in [1, cap] with the best value
after the charge, as in overpromise_listener, not stop_listener's count of
what the clue points at.

**The STOP listener** defaults to the strict one
(cache/listener_gbt_stop_guess.txt, trained on gpt-oss's "guess" group): the
comfort arena's guesser stops when that group runs out, and the strict
models were the comfortable end of the frontier. `stop_model_path` overrides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codenames.game import Role
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.stop_listener import StopListenerSpymaster

STRICT_STOP_MODEL = "listener_gbt_stop_guess.txt"


def turn_values_promise(S: np.ndarray, n_own: int, roles: list[Role], V: np.ndarray, K: int,
                        charge: float) -> np.ndarray:
    """P(win) after the turn minus `charge` per promised word not attempted,
    for numbers 1..K. As stop_listener.turn_values, with the STOP ending of a
    clue for k after j picks charged `charge` * (k - j)."""
    n = S.shape[1] - 1
    b = sum(r == Role.OPPONENT for r in roles)
    lam = np.exp(S - S.max(axis=1, keepdims=True))

    def after(left: int, b_left: int) -> float:
        return 1.0 if left == 0 else 1.0 - V[b_left, left]

    end_bad = {}
    for left in range(1, n_own + 1):
        end_bad[left] = np.array([0.0 if r == Role.ASSASSIN else
                                  (0.0 if b == 1 else after(left, b - 1)) if r == Role.OPPONENT else
                                  after(left, b) for r in roles])
    out = np.zeros(K)
    for k in range(1, K + 1):
        memo: dict = {}

        def value(found: frozenset) -> float:
            if found in memo:
                return memo[found]
            j, left = len(found), n_own - len(found)
            if left == 0 or j == k:
                memo[found] = after(left, b)
                return memo[found]
            own_left = [i for i in range(n_own) if i not in found]
            w_own, w_bad, w_stop = lam[j, own_left], lam[j, n_own:n], lam[j, n]
            z = w_own.sum() + w_bad.sum() + w_stop
            v = (w_bad @ end_bad[left] + w_stop * (after(left, b) - charge * (k - j))) / z
            for i, w in zip(own_left, w_own):
                v += w / z * value(found | {i})
            memo[found] = v
            return v

        out[k - 1] = value(frozenset())
    return out


class WinProbOverpromiseListenerSpymaster(StopListenerSpymaster):
    def __init__(self, *args, promise_cost: float = 0.5, cache_dir: Path = DEFAULT_CACHE_DIR,
                 stop_model_path: Path | None = None, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir,
                         stop_model_path=Path(stop_model_path or cache_dir / STRICT_STOP_MODEL), **kwargs)
        self.promise_cost = float(promise_cost)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return super().model_files({**params, "stop_model_path": params.get("stop_model_path")
                                    or cache_dir / STRICT_STOP_MODEL})

    # The play server's label: P(win) after the turn minus the overpromise cost.
    value_kind = "win_charged"

    def picks_needed(self, cap: int) -> int:
        return cap

    def _charge(self, n_own: int, roles: list[Role]) -> float:
        """promise_cost words, in win probability at this score."""
        b = sum(r == Role.OPPONENT for r in roles)
        return self.promise_cost * max(self.value.one_word(n_own, b), 0.0)

    def _values(self, S: np.ndarray, n_own: int, roles: list[Role], K: int) -> np.ndarray:
        return turn_values_promise(S, n_own, roles, self.value.V, K, self._charge(n_own, roles))

    def turn_value(self, S: np.ndarray, n_own: int, roles: list[Role], k: int) -> float:
        """P(win) after a turn with number k, minus the charge for unattempted promises."""
        return float(self._values(S, n_own, roles, k)[-1])

    def choose(self, S: np.ndarray, n_own: int, roles: list[Role], cap: int) -> tuple[int, float]:
        v = self._values(S[:cap], n_own, roles, cap)
        return int(v.argmax()) + 1, float(v.max())
