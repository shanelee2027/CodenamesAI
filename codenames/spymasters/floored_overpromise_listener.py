"""margin_value_overpromise_listener with a floor under the overpromise cost
(docs/versions/floored_overpromise_listener.md).

**Why.** The cost per promised word the guesser does not attempt is
promise_cost * one_word(a, b): what one more own word is worth now, in win
probability. one_word falls to near 0 where one word cannot change the game,
which is mostly when the side to move is two or more words behind (e.g.
one_word(5, 2) = 0.006, one_word(7, 3) = 0.007 on the margin V). There the
cost disappears, and the P(win) objective, which already favours long shots
when behind, can promise numbers its own guesser will not attempt at no
charge.

**What changed: only the charge.** It is promise_cost * max(one_word(a, b),
charge_floor). A constant charge was the other option; it would have to sit
near the middle-game value (~0.15) to matter, which is seven times the
opening's 0.020 and so makes the opening clue smaller still. The floor
leaves every position where one_word is above it as it was.

**The floor, 0.05** (docs/versions/floored_overpromise_listener.md; free
check at λ = 1, 12 boards per score). With the opponent on its last word and
4 of ours left, no floor promised 4 on 10 boards, mostly with clues that do
not connect them (BLANK, GOVERNMENT, FICTIONAL); 0.02 still did on 3 and
0.05 on none. It also charges the opening, where one_word is 0.020, a
little more: mean number 3.08 -> 3.00 there (0.1 would give 2.58).
"""

from __future__ import annotations

from codenames.game import Role
from codenames.spymasters.margin_value_overpromise_listener import MarginValueOverpromiseListenerSpymaster


class FlooredOverpromiseListenerSpymaster(MarginValueOverpromiseListenerSpymaster):
    def __init__(self, *args, charge_floor: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.charge_floor = float(charge_floor)

    def _charge(self, n_own: int, roles: list[Role]) -> float:
        """promise_cost words, in win probability at this score, with one word
        worth at least `charge_floor`."""
        b = sum(r == Role.OPPONENT for r in roles)
        return self.promise_cost * max(self.value.one_word(n_own, b), self.charge_floor)
