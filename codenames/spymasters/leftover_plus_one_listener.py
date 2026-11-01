"""online_vocab_listener with the guesser's +1 spent on the previous clue's
leftover (docs/versions/leftover_plus_one_listener.md).

**Why.** Codenames allows one guess beyond the number, and a human spends it
on a word from an earlier clue that the team did not find. Every model so
far values a turn as if the +1 did not exist, so a word left over from the
last clue is worth nothing to it until it is clued again, and it re-clues
leftovers it could have left to the +1 (docs/log.md, the history probes).

**The rule** (the user's, docs/log.md, "The leftover +1"):
1. Only our previous clue counts.
2. It left something only if its turn was cut short by a wrong guess.
3. Unfound: its number, minus the own words that turn found, minus the own
   words the opponent has revealed since that the clue was meant for (the
   guesser sees "your word" turn up and counts it as found). Nothing left
   to go back to at 0.
4. If this turn ends with nothing wrong and own words still to find, one
   more guess goes back to the previous clue.

**The value.** online_vocab_listener's turn (the exact recursion over own
words found, with the floored overpromise charge), with one change: where
the turn ends cleanly -- at STOP, or at the number with every guess ours --
the value is no longer P(win) after the turn but the +1's expectation:
    q * [sum_w p(w) * value after w  +  p(STOP) * value after the turn]
      + (1 - q) * value after the turn,
where p is the STOP listener's next pick for the previous clue (its pick
j + 1 after the j guesses it got) over the words on the board and STOP,
renormalised without the own words this turn found. An own word found by
the +1 ends the turn one word better off; a bad one ends it as a wrong guess
does; STOP is the guesser passing because the old clue points at nothing
clear any more. So the vagueness check is in p: a leftover the listener
rates low carries little weight, and the replay
(scripts/tools/replay_leftover_plus_one.py) found that such leftovers are
mostly not ours. q = `plus_one_rate`, 1 by default.

**Which own words the opponent revealed were the clue's** is not known to
the spymaster. Each counts as the clue's with probability p0(w): the STOP
listener's next pick for the previous clue over the board as our previous
turn left it. q is multiplied by the chance that unfound is still above 0.

**History** comes in through `TurnContext.history`. Without it (the arena's
game loop, a first turn) the model is online_vocab_listener exactly.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from codenames.game import Role
from codenames.spymasters.base import PastTurn, TurnContext
from codenames.spymasters.online_vocab_listener import OnlineVocabListenerSpymaster


def turn_values_leftover(S: np.ndarray, n_own: int, roles: list[Role], V: np.ndarray, K: int,
                         charge: float, L: np.ndarray, q: float) -> np.ndarray:
    """win_prob_overpromise_listener.turn_values_promise with the leftover
    +1 at every clean ending. L is (n_words + 1,): the +1's softmax over the
    words in S's order and STOP; q the chance the +1 is taken."""
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
    tails: dict = {}

    def tail(found: frozenset) -> float:
        """The clean ending's value with the +1, `found` own words found this turn."""
        if found not in tails:
            left = n_own - len(found)
            own_left = [i for i in range(n_own) if i not in found]
            l_own, l_bad, l_stop = L[own_left], L[n_own:n], L[n]
            z = l_own.sum() + l_bad.sum() + l_stop
            t = (l_own.sum() * after(left - 1, b) + l_bad @ end_bad[left] + l_stop * after(left, b)) / z
            tails[found] = q * t + (1 - q) * after(left, b)
        return tails[found]

    out = np.zeros(K)
    for k in range(1, K + 1):
        memo: dict = {}

        def value(found: frozenset) -> float:
            if found in memo:
                return memo[found]
            j, left = len(found), n_own - len(found)
            if left == 0:
                memo[found] = 1.0
                return 1.0
            if j == k:
                memo[found] = tail(found)
                return memo[found]
            own_left = [i for i in range(n_own) if i not in found]
            w_own, w_bad, w_stop = lam[j, own_left], lam[j, n_own:n], lam[j, n]
            z = w_own.sum() + w_bad.sum() + w_stop
            v = (w_bad @ end_bad[left] + w_stop * (tail(found) - charge * (k - j))) / z
            for i, w in zip(own_left, w_own):
                v += w / z * value(found | {i})
            memo[found] = v
            return v

        out[k - 1] = value(frozenset())
    return out


def still_unfound(remaining: int, p0: list[float]) -> float:
    """P(fewer than `remaining` of the opponent-revealed own words were the
    clue's), each independently with probability p0."""
    dist = np.zeros(len(p0) + 1)
    dist[0] = 1.0
    for p in p0:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[:remaining].sum())


class LeftoverPlusOneListenerSpymaster(OnlineVocabListenerSpymaster):
    uses_history = True                 # the play server passes the game's turns to clue_value too

    def __init__(self, *args, plus_one_rate: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.plus_one_rate = float(plus_one_rate)
        self._leftover: tuple[np.ndarray, float] | None = None

    def leftover(self, board, history: tuple[PastTurn, ...], sims) -> tuple[np.ndarray, float] | None:
        """(L, q) for this position, L over `_board(board)`'s words and STOP,
        or None when there is nothing to go back to."""
        prev_i = next((i for i in range(len(history) - 1, -1, -1) if history[i].ours), None)
        if prev_i is None:
            return None
        prev = history[prev_i]
        guesses = [g for g in prev.guesses if g in board.words]
        if not guesses or not prev.clue or board.role_of(guesses[-1]) == Role.OWN:
            return None
        remaining = int(prev.number) - sum(board.role_of(g) == Role.OWN for g in guesses)
        if remaining <= 0:
            return None
        own, words, _ = self._board(board)
        if not own:
            return None
        j = len(guesses)
        s = self.stop_scores([prev.clue], words, j + 1, sims).get(0)
        if s is None:
            return None
        L = np.exp(s[j] - s[j].max())
        L /= L.sum()
        # Own words the opponent turned over since: were they the clue's?
        since = [g for t in history[prev_i + 1:] if not t.ours for g in t.guesses if g in board.words]
        taken = [g for g in since if board.role_of(g) == Role.OWN]
        q = self.plus_one_rate
        if taken:
            then = words + since
            s0 = self.stop_scores([prev.clue], then, j + 1, sims).get(0)
            if s0 is not None:
                p = np.exp(s0[j] - s0[j].max())
                p /= p.sum()
                q *= still_unfound(remaining, [float(p[then.index(g)]) for g in taken])
        return L, q

    @contextmanager
    def history(self, board, history: tuple[PastTurn, ...], sims):
        """The search below values turns with this position's leftover."""
        self._leftover = self.leftover(board, tuple(history), sims) if history else None
        try:
            yield
        finally:
            self._leftover = None

    def top_clues(self, ctx: TurnContext, sims, k: int):
        with self.history(ctx.board, ctx.history, sims):
            return super().top_clues(ctx, sims, k)

    def clue_value(self, board, clue: str, number: int, sims, history: tuple[PastTurn, ...] = ()) -> float | None:
        with self.history(board, history, sims):
            return super().clue_value(board, clue, number, sims)

    def _values(self, S: np.ndarray, n_own: int, roles: list[Role], K: int) -> np.ndarray:
        if self._leftover is None:
            return super()._values(S, n_own, roles, K)
        L, q = self._leftover
        return turn_values_leftover(S, n_own, roles, self.value.V, K, self._charge(n_own, roles), L, q)
