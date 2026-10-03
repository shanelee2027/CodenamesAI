"""win_prob_listener that looks one move ahead: what does the incumbent do on
the board our clue leaves? (docs/versions/reply_lookahead_listener.md)

**Why.** V(a, b) values the position after our turn by the incumbent's
average win probability at that score, the same for every board. But which
words our clue removes changes what the opponent can do next. Our unrevealed
words are its danger words: if our "Nile" is still up, its "river" clue is
risky; if our clue gets Nile found, we have unlocked its river cluster. A
human spymaster weighs this, and V cannot see it.

**The lookahead,** on top of win_prob_listener's search:
1. The search ranks clues by P(win) with V as before. The best `top_clues`
   clues are scored at every number, and the best `candidates` (clue, number)
   pairs go on.
2. Each pair's turn is enumerated under the frozen listener (exact
   Plackett-Luce picks, stopping at the first word that is not ours or after
   the number): which own words are found and which word ends the turn. Each
   outcome is an after-board.
3. On each probable after-board (P >= `min_outcome`, at most `max_outcomes`
   per pair) the incumbent's reply is worked out, and our value there is
   1 - its P(win) with that clue, minus the lookahead's mean offset from V
   at that score (scripts/data/build_reply_offset.py): V sets the level at
   each score, the lookahead only how this board differs from it. The
   other outcomes keep V's value. A pair's
   value is win_prob_listener's, plus P(outcome) x (lookahead value - V's
   value) summed over the probable outcomes.
4. The pair with the highest value is played.

**The incumbent's reply, approximated cheaply.** Its full search runs once per
move, on the current board from its side. Its best `opponent_clues` clues are
kept, with its listener's score for every word. On an after-board, the
revealed words are dropped from those scores, and the incumbent picks among
those clues by its own objective (expected words minus role costs). Its
P(win) with that pick is computed from our listener's scores for its words,
with V at the end of its turn. What this ignores:
- features that depend on the whole board (a word's rank among those left,
  the candidate count, cohesion) shift a little when words are revealed;
- a clue outside its current top `opponent_clues` cannot become its reply.
Both are measured against full re-searches (docs/log.md, "reply lookahead").

**Parameters:** win_prob_listener's (frozen turn model and no calibration
only), plus `opponent_model_path` (the incumbent's booster by default),
`top_clues`, `candidates`, `opponent_clues`, `min_outcome`, `max_outcomes`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codenames.board import Board, OpponentBoardView, clue_number_cap
from codenames.game import Role
from codenames.pl_reward import gain_and_penalty
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.conceptnet_listener import MODEL as CONCEPTNET_MODEL
from codenames.spymasters.learned_listener import LearnedListenerSpymaster
from codenames.spymasters.win_prob_listener import WinProbListenerSpymaster
from codenames.win_value import win_probability

INCUMBENT = "listener_gbt.txt"
OFFSET = "reply_offset.npz"
PRUNE = 1e-4            # enumeration branches below this probability are left to V


def other_side(board):
    """The same board from the other team's side (shared revealed state)."""
    return board._board if isinstance(board, OpponentBoardView) else OpponentBoardView(board)


def turn_outcomes(s: np.ndarray, n_own: int, k: int, prune: float = PRUNE) -> dict[tuple, float]:
    """A turn under frozen Plackett-Luce on scores `s` (own words first): the
    guesser picks in order with P proportional to exp(s) among the words
    left, and the turn ends after k own words, or at the first word that is
    not ours. Returns {(own words found, ending word index or None): P}.
    Branches below `prune` are dropped, so the values sum to a little under 1."""
    lam = np.exp(s - s.max())
    out: dict[tuple, float] = {}

    def rec(left: np.ndarray, found: tuple, p: float) -> None:
        idx = np.flatnonzero(left)
        q = p * lam[idx] / lam[idx].sum()
        for i, qi in zip(idx.tolist(), q.tolist()):
            if qi < prune:
                continue
            if i < n_own:
                f = found + (i,)
                if len(f) == k or len(f) == n_own:
                    key = (frozenset(f), None)
                    out[key] = out.get(key, 0.0) + qi
                else:
                    nxt = left.copy()
                    nxt[i] = False
                    rec(nxt, f, qi)
            else:
                key = (frozenset(found), i)
                out[key] = out.get(key, 0.0) + qi

    rec(np.ones(len(s), dtype=bool), (), 1.0)
    return out


class ReplyLookaheadListenerSpymaster(WinProbListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None,
                 opponent_model_path: Path | None = None, top_clues: int = 8, candidates: int = 6,
                 opponent_clues: int = 30, min_outcome: float = 0.01, max_outcomes: int = 12,
                 reply_offset_path: Path | str | None = None, **kwargs):
        model_path = Path(model_path) if model_path else Path(cache_dir) / CONCEPTNET_MODEL
        super().__init__(*args, cache_dir=cache_dir, model_path=model_path, **kwargs)
        if self.turn_model != "frozen" or self.calibration is not None:
            raise ValueError("reply_lookahead_listener enumerates turns under the frozen model, uncalibrated")
        if self.outside_n:
            raise ValueError("reply_lookahead_listener does not model the outside option")
        opp_path = Path(opponent_model_path) if opponent_model_path else Path(cache_dir) / INCUMBENT
        self.opp = LearnedListenerSpymaster(cache_dir=cache_dir, model_path=opp_path, clue_stats=self.clue_stats)
        # With the same booster on both sides, the incumbent's scores are ours.
        self._same_listener = opp_path.resolve() == model_path.resolve()
        self.n_top_clues, self.n_candidates, self.opponent_clues = top_clues, candidates, opponent_clues
        self.min_outcome, self.max_outcomes = min_outcome, max_outcomes
        # The lookahead's mean offset from V at each score
        # (scripts/data/build_reply_offset.py); "" means none, for building it.
        if reply_offset_path == "":
            self.offset = np.zeros_like(self.value.V)
        else:
            self.offset = np.load(Path(reply_offset_path) if reply_offset_path else Path(cache_dir) / OFFSET)["offset"]

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        files = [*super().model_files(params), Path(params.get("opponent_model_path") or cache_dir / INCUMBENT)]
        if params.get("reply_offset_path") != "":
            files.append(Path(params.get("reply_offset_path") or cache_dir / OFFSET))
        return files

    # -- the incumbent's side ------------------------------------------------

    def _scores(self, spymaster, clues: list[str], words: list[str], number: int, sims) -> tuple[list[int], np.ndarray]:
        """Listener scores (rows that have features, (n, len(words)))."""
        rows, keep = [], []
        for j, c in enumerate(clues):
            f = spymaster._listener_features(c, words, number, sims)
            if f is not None:
                rows.append(f)
                keep.append(j)
        if not rows:
            return [], np.zeros((0, len(words)))
        flat = spymaster.bundle.booster.predict(np.vstack(rows), raw_score=True)
        return keep, np.asarray(flat, dtype=np.float64).reshape(len(keep), len(words))

    def _opponent(self, board, sims) -> dict | None:
        """The incumbent's view of the current board, searched once."""
        view = other_side(board)
        own = view.words_by_role(Role.OWN, unrevealed_only=True)
        bad = {r: view.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        _, scores, _ = self.opp._score_all_clues(view, sims)
        finite = np.flatnonzero(np.isfinite(scores))
        if not own or finite.size == 0:
            return None
        top = finite[np.argsort(-scores[finite])[: self.opponent_clues]]
        clues = [sims.clue_words[i] for i in top]
        number = clue_number_cap(len(own), self.opp.max_number)
        keep, S_inc = self._scores(self.opp, clues, words, number, sims)
        if not keep:
            return None
        if self._same_listener:
            S_val = S_inc
        else:
            k2, S_val = self._scores(self, [clues[j] for j in keep], words, number, sims)
            if len(k2) != len(keep):
                return None
        return {"words": words, "n_own": len(own), "roles": roles, "S_inc": S_inc, "S_val": S_val,
                "clues": [clues[j] for j in keep]}

    def _reply_win(self, opp: dict, removed: frozenset) -> float:
        """The incumbent's P(win) with the clue it would give once `removed`
        are revealed."""
        return self._reply(opp, removed)[2]

    def _reply(self, opp: dict, removed: frozenset) -> tuple[str | None, int, float]:
        """(the incumbent's clue, its number, its P(win)) once `removed` are
        revealed."""
        words, n_own = opp["words"], opp["n_own"]
        own_cols = [i for i in range(n_own) if words[i] not in removed]
        bad_cols = [i for i in range(n_own, len(words)) if words[i] not in removed]
        if not own_cols:
            return None, 0, 1.0
        roles = [opp["roles"][i - n_own] for i in bad_cols]
        costs = np.array([self.opp.costs[r] for r in roles], dtype=np.float64)
        K = clue_number_cap(len(own_cols), self.opp.max_number)
        S_inc, S_val = opp["S_inc"], opp["S_val"]
        vals, _ = self.opp._clue_values(S_inc[:, own_cols], S_inc[:, bad_cols], roles, costs, K, None)
        m, kk = np.unravel_index(int(np.argmax(vals)), vals.shape)
        so, sb = S_val[m:m + 1][:, own_cols], S_val[m:m + 1][:, bad_cols]
        n_their_opp = sum(r == Role.OPPONENT for r in roles)
        wp = win_probability(lambda c: gain_and_penalty(so, sb, c, K), roles, K, len(own_cols), n_their_opp, self.value)
        return opp["clues"][m], int(kk) + 1, float(wp[0, kk])

    # -- our side ------------------------------------------------------------

    def _correction(self, s: np.ndarray, k: int, words: list[str], n_own: int, roles: list[Role],
                    opp: dict, cache: dict) -> float:
        """Sum over the probable outcomes of P x (lookahead value - V's value)."""
        a = n_own
        b = sum(r == Role.OPPONENT for r in roles)
        V = self.value.V
        outcomes = sorted(turn_outcomes(s, n_own, k).items(), key=lambda t: -t[1])
        total = 0.0
        for (found, end), p in outcomes[: self.max_outcomes]:
            if p < self.min_outcome:
                break
            left = a - len(found)
            role = None if end is None else roles[end - n_own]
            if left == 0 or role == Role.ASSASSIN or (role == Role.OPPONENT and b == 1):
                continue                                     # the game is over: V's value is exact
            b_after = b - 1 if role == Role.OPPONENT else b
            w_v = 1.0 - V[b_after, left]
            removed = frozenset([words[i] for i in found] + ([] if end is None else [words[end]]))
            if removed not in cache:
                cache[removed] = self._reply_win(opp, removed)
            total += p * ((1.0 - cache[removed]) - self.offset[b_after, left] - w_v)
        return total

    def _score_all_clues(self, board: Board | OpponentBoardView, sims):
        best_n, scores, margin = super()._score_all_clues(board, sims)
        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        finite = np.flatnonzero(np.isfinite(scores))
        if not own or finite.size == 0:
            return best_n, scores, margin
        bad = {r: board.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        n_own, K = len(own), clue_number_cap(len(own), self.max_number)

        top = finite[np.argsort(-scores[finite])[: self.n_top_clues]]
        keep, S = self._scores(self, [sims.clue_words[i] for i in top], words, K, sims)
        opp = self._opponent(board, sims)
        if not keep or opp is None:
            return best_n, scores, margin
        top = top[keep]
        costs = np.array([self.costs[r] for r in roles], dtype=np.float64)
        base, _ = self._clue_values(S[:, :n_own], S[:, n_own:], roles, costs, K, None, words=words)   # (T, K)
        order = np.argsort(-base, axis=None)[: self.n_candidates]
        cache: dict = {}
        best: dict[int, tuple[float, int]] = {}
        for flat in order.tolist():
            t, m = divmod(flat, base.shape[1])
            v = base[t, m] + self._correction(S[t], m + 1, words, n_own, roles, opp, cache)
            ci = int(top[t])
            if ci not in best or v > best[ci][0]:
                best[ci] = (v, m + 1)
        # Only the evaluated pairs compete: everything else drops below them,
        # keeping its order (win probabilities are within [0, 1]).
        scores = np.where(np.isfinite(scores), scores - 10.0, scores).astype(scores.dtype)
        best_n = best_n.copy()
        for ci, (v, k) in best.items():
            scores[ci], best_n[ci] = v, k
        return best_n, scores, margin
