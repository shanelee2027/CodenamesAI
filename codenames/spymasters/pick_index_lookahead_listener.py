"""reply_lookahead_listener with the assoc booster and a pick-index model of
the turn (docs/versions/pick_index_lookahead_listener.md).

**What changes from reply_lookahead_listener.** That model enumerates our turn
under the frozen listener: one score vector, the guesser picking among the
words left by softmax of the same scores at every pick. The frozen listener
is overconfident at picks 2+. Here the turn is enumerated under the
pick-index booster (scripts/pipeline/train_pick_index_listener.py,
`cache/listener_gbt_pick_indexassoc_depth9.txt`). It reads the assoc
booster's 56 features plus the pick number j and j - k, so at pick j the
guesser picks by softmax of S_j, its own score vector for that pick and
that clue number. Its Sonnet 5.5 R² is 0.3374 against the frozen assoc
booster's 0.3293, all of it at picks 2+ (docs/log.md, "Progress notebook").
Nothing is done to the scores after the booster: the pick-index booster
learned its own sharpness at each pick.

**Where.** win_prob_listener's search ranks the shortlist with the frozen
assoc booster, as before. Its best `top_clues` (8) clues are scored by the
pick index at every number k: k booster rows per word, one per pick. Their
P(win) is computed by enumerating each turn exactly under those scores, and
the best `candidates` (clue, number) pairs get the reply lookahead. Searching
the whole shortlist under the pick index instead picks the same clue and
number on 99 of 100 positions and costs about 5x the time
(docs/log.md, "pick_index_lookahead_listener").

**Kept from reply_lookahead_listener.** The incumbent's reply: its clue
chosen by its own listener and objective, and its turn played out under our
frozen booster with V at the end. The offset from V at each score is rebuilt
under this turn model (`cache/reply_offset_pick_index.npz`,
scripts/data/build_reply_offset.py --spymaster pick_index_lookahead_listener).
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from codenames.board import Board, OpponentBoardView, clue_number_cap
from codenames.game import Role
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.reply_lookahead_listener import PRUNE, ReplyLookaheadListenerSpymaster
from codenames.spymasters.win_prob_listener import WinProbListenerSpymaster

ASSOC_MODEL = "listener_gbt_assoc_features.txt"
PICK_MODEL = "listener_gbt_pick_indexassoc_depth9.txt"
PICK_OFFSET = "reply_offset_pick_index.npz"
PICK_COLUMNS = ["x", "x_minus_k"]       # the pick number (1-based) and pick number - k


def pick_turn_outcomes(S: np.ndarray, n_own: int, k: int, prune: float = PRUNE) -> dict[tuple, float]:
    """reply_lookahead_listener.turn_outcomes with a score vector per pick:
    the guesser's (j+1)-th pick is by softmax of S[j] over the words left
    (own words first). The turn ends after k own words or at the first word
    that is not ours. {(own words found, ending word index or None): P},
    branches below `prune` dropped."""
    lam = np.exp(S - S.max(axis=1, keepdims=True))
    out: dict[tuple, float] = {}

    def rec(left: np.ndarray, found: tuple, p: float) -> None:
        idx = np.flatnonzero(left)
        w = lam[len(found)][idx]
        for i, qi in zip(idx.tolist(), (p * w / w.sum()).tolist()):
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

    rec(np.ones(S.shape[1], dtype=bool), (), 1.0)
    return out


class PickIndexLookaheadListenerSpymaster(ReplyLookaheadListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None,
                 pick_model_path: Path | None = None, reply_offset_path: Path | str | None = None, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir, model_path=Path(model_path or cache_dir / ASSOC_MODEL),
                         reply_offset_path=cache_dir / PICK_OFFSET if reply_offset_path is None else reply_offset_path,
                         **kwargs)
        self.pick_booster = lgb.Booster(model_file=str(pick_model_path or cache_dir / PICK_MODEL))
        names = self.bundle.booster.feature_name()
        if self.pick_booster.feature_name() != names + PICK_COLUMNS:
            raise ValueError("the pick-index booster must read the listener's columns plus " + ", ".join(PICK_COLUMNS))

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        p = {**params, "model_path": params.get("model_path") or cache_dir / ASSOC_MODEL}
        if params.get("reply_offset_path") is None:
            p["reply_offset_path"] = cache_dir / PICK_OFFSET
        return [*super().model_files(p), cache_dir / "assoc_sims.npz",
                Path(params.get("pick_model_path") or cache_dir / PICK_MODEL)]

    # -- our turn under the pick index ---------------------------------------

    def pick_scores(self, clues: list[str], words: list[str], K: int, sims) -> dict[tuple[int, int], np.ndarray]:
        """{(clue index into `clues`, k): (k, len(words)) scores, row j for
        pick j+1}, for every clue with features and k = 1..K, in one booster
        call (per-call overhead dominates 25-row calls)."""
        rows, index = [], []
        for t, c in enumerate(clues):
            for k in range(1, K + 1):
                f = self._listener_features(c, words, k, sims)
                if f is None:
                    break
                for j in range(k):
                    rows.append(np.hstack([f, np.tile([j + 1.0, j + 1.0 - k], (len(f), 1))]))
                index.append((t, k))
        if not rows:
            return {}
        flat = np.asarray(self.pick_booster.predict(np.vstack(rows), raw_score=True), dtype=np.float64)
        out, at, n = {}, 0, len(words)
        for t, k in index:
            out[(t, k)] = flat[at:at + k * n].reshape(k, n)
            at += k * n
        return out

    def turn_value(self, outcomes: dict[tuple, float], n_own: int, roles: list[Role]) -> float:
        """P(win) after the turn, V at each ending, as win_prob_listener:
        all own words found wins, the assassin or the opponent's last word
        loses, otherwise 1 - V(their words left, ours left). Normalised over
        the enumerated outcomes, so pruned branches do not count as losses."""
        b = sum(r == Role.OPPONENT for r in roles)
        V, total, mass = self.value.V, 0.0, 0.0
        for (found, end), p in outcomes.items():
            mass += p
            left = n_own - len(found)
            role = None if end is None else roles[end - n_own]
            if left == 0:
                total += p
            elif role == Role.ASSASSIN or (role == Role.OPPONENT and b == 1):
                continue
            else:
                total += p * (1.0 - V[b - (role == Role.OPPONENT), left])
        return total / mass if mass > 0 else 0.0

    def candidate_turns(self, board, sims, scores: np.ndarray) -> dict | None:
        """The top clues of win_prob's frozen search, scored by the pick index
        at every number: {"words", "roles", "n_own", "pairs": [(clue index,
        k, P(win), outcomes)]}. Shared with build_reply_offset.py."""
        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        finite = np.flatnonzero(np.isfinite(scores))
        if not own or finite.size == 0:
            return None
        bad = {r: board.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        n_own, K = len(own), clue_number_cap(len(own), self.max_number)
        top = finite[np.argsort(-scores[finite])[: self.n_top_clues]]
        S = self.pick_scores([sims.clue_words[i] for i in top], words, K, sims)
        pairs = []
        for (t, k), s in S.items():
            o = pick_turn_outcomes(s, n_own, k)
            pairs.append((int(top[t]), k, self.turn_value(o, n_own, roles), o))
        return {"words": words, "roles": roles, "n_own": n_own, "pairs": pairs} if pairs else None

    def _outcome_correction(self, outcomes: dict[tuple, float], words: list[str], n_own: int,
                            roles: list[Role], opp: dict, cache: dict) -> float:
        """reply_lookahead_listener._correction on given outcomes."""
        b = sum(r == Role.OPPONENT for r in roles)
        V, total = self.value.V, 0.0
        for (found, end), p in sorted(outcomes.items(), key=lambda t: -t[1])[: self.max_outcomes]:
            if p < self.min_outcome:
                break
            left = n_own - len(found)
            role = None if end is None else roles[end - n_own]
            if left == 0 or role == Role.ASSASSIN or (role == Role.OPPONENT and b == 1):
                continue                                     # the game is over: V's value is exact
            b_after = b - 1 if role == Role.OPPONENT else b
            removed = frozenset([words[i] for i in found] + ([] if end is None else [words[end]]))
            if removed not in cache:
                cache[removed] = self._reply_win(opp, removed)
            total += p * ((1.0 - cache[removed]) - self.offset[b_after, left] - (1.0 - V[b_after, left]))
        return total

    def _score_all_clues(self, board: Board | OpponentBoardView, sims):
        # win_prob_listener's frozen search ranks the shortlist; the lookahead
        # stage below replaces reply_lookahead_listener's.
        best_n, scores, margin = WinProbListenerSpymaster._score_all_clues(self, board, sims)
        cand = self.candidate_turns(board, sims, scores)
        opp = self._opponent(board, sims) if cand else None
        if cand is None or opp is None:
            return best_n, scores, margin
        cache: dict = {}
        best: dict[int, tuple[float, int]] = {}
        for ci, k, v, o in sorted(cand["pairs"], key=lambda t: -t[2])[: self.n_candidates]:
            v += self._outcome_correction(o, cand["words"], cand["n_own"], cand["roles"], opp, cache)
            if ci not in best or v > best[ci][0]:
                best[ci] = (v, k)
        # Only the evaluated pairs compete; the rest keep their order below them.
        scores = np.where(np.isfinite(scores), scores - 10.0, scores).astype(scores.dtype)
        best_n = best_n.copy()
        for ci, (v, k) in best.items():
            scores[ci], best_n[ci] = v, k
        return best_n, scores, margin

    def clue_value(self, board, clue: str, number: int, sims) -> float | None:
        """P(win) after this clue's turn under the pick index, with V at each
        ending (no lookahead): the play server's explanation."""
        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        bad = {r: board.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        S = self.pick_scores([clue], words, number, sims).get((0, number))
        if S is None or not own:
            return None
        return self.turn_value(pick_turn_outcomes(S, len(own), number), len(own), roles)
