"""Score a clue by the probability of winning after the turn it produces.

**The objective.** Every listener spymaster used to maximise, per clue and
number k,

    expected own words  -  expected role cost   (costs 0.2 / 1 / 10)

which prices one turn and cannot see tempo. Here the same listener gives the
distribution of how the turn ends, and each ending leaves a score position
whose value comes from V (scripts/data/build_win_value.py):

    P(win | clue, k) =  sum over j < k of P(j own words, then word w ends the turn) * W(ending)
                      + P(all k own words found) * W(stop after k)

- **W, with a own words and b opponent words left now:**
  - the turn stops after k own words: 1 if k = a (the board is cleared),
    else 1 - V(b, a - k), since the opponent is then to move;
  - it ends on a neutral word after j: 1 - V(b, a - j);
  - it ends on an opponent word: 1 - V(b - 1, a - j), or 0 if that was the
    opponent's last word;
  - it ends on the assassin: 0.
- **Where the probabilities come from.** They come from the spymaster's own
  reward model through its `_gain_and_penalty`, called with indicator costs:
  1 on the opponent words (then on the assassin), 0 elsewhere. That call
  returns cumulative sums over k:
  - `gain[k] - gain[k-1]` = P(the guesser gets its k-th own word);
  - `penalty[k] - penalty[k-1]` = P(it got exactly k-1, then hit that role).

  So the one objective works unchanged over any reward model: frozen
  Plackett-Luce scores, per-pick temperatures, or the within-turn model.
- **An outside option,** when on, is a free end of turn and counts as
  neutral.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from codenames.game import Role


class WinValue:
    """V(a, b) = P(the side to move wins | it has a own words left, the other
    side has b), from cache/win_value.npz."""

    def __init__(self, path: Path):
        self.V = np.load(path)["V"]

    def after(self, a: int, b: int) -> dict[str, np.ndarray]:
        """Our win probability for each way a turn can end, starting from a
        own and b opponent words left. Arrays indexed by j = own words found
        (0..a); `stop[j]` is also the value of stopping after j."""
        V = self.V
        j = np.arange(a + 1)
        left = a - j
        stop = np.where(left == 0, 1.0, 1.0 - V[b, np.maximum(left, 1)])
        neutral = stop.copy()                        # the turn ends, nothing else changes
        opponent = np.zeros(a + 1) if b == 1 else np.where(left == 0, 1.0, 1.0 - V[b - 1, np.maximum(left, 1)])
        return {"stop": stop, "neutral": neutral, "opponent": opponent}

    def one_word(self, a: int, b: int) -> float:
        """What one more own word this turn is worth now, in win probability:
        stopping after 1 against ending the turn with 0. The k=1 tie-break
        tolerance is in own words and is scaled by this."""
        w = self.after(a, b)
        return float(w["stop"][1] - w["neutral"][0])


def win_probability(gain_and_penalty: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                    roles: list[Role], max_k: int, n_own: int, n_opponent: int,
                    value: WinValue) -> np.ndarray:
    """`(n_clues, max_k)`: P(win) for each clue at k = 1..max_k.

    `gain_and_penalty(costs)` runs the reward model on this board's scores
    with the given per-bad-word costs (the roles in `roles` order) and
    returns its cumulative `(gain, penalty)`."""
    roles_arr = np.array([r.value for r in roles])
    ind = {r: (roles_arr == r.value).astype(np.float64) for r in (Role.OPPONENT, Role.ASSASSIN)}
    gain, p_opp = gain_and_penalty(ind[Role.OPPONENT])
    _, p_ass = gain_and_penalty(ind[Role.ASSASSIN])
    gain, p_opp, p_ass = (np.asarray(x, dtype=np.float64) for x in (gain, p_opp, p_ass))
    C = gain.shape[0]
    zero = np.zeros((C, 1))
    # reach[:, j] = P(at least j own words found), j = 0..max_k
    reach = np.clip(np.concatenate([np.ones((C, 1)), np.diff(np.concatenate([zero, gain], 1), axis=1)], 1), 0, 1)
    # end_r[:, j] = P(exactly j own words, then a word of role r), j = 0..max_k-1
    end_opp = np.clip(np.diff(np.concatenate([zero, p_opp], 1), axis=1), 0, 1)
    end_ass = np.clip(np.diff(np.concatenate([zero, p_ass], 1), axis=1), 0, 1)
    exactly = np.clip(reach[:, :-1] - reach[:, 1:], 0, 1)             # P(N = j), j < max_k
    end_neu = np.clip(exactly - end_opp - end_ass, 0, 1)                # neutral, or the outside option

    # W per ending, (a + 1,) from the score alone, or (n_clues, >= max_k + 1)
    # when the after-board differs by clue (codenames/board_value.py).
    w = value.after(n_own, n_opponent)
    ended = end_neu * w["neutral"][..., :max_k] + end_opp * w["opponent"][..., :max_k]   # the assassin is worth 0
    # At k: every ending with j < k own words, plus finding all k and stopping.
    return np.cumsum(ended, axis=1) + reach[:, 1:] * w["stop"][..., 1:max_k + 1]
