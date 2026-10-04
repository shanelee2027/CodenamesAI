"""win_prob_listener with a guesser that can end its turn
(docs/versions/stop_listener.md).

**Why.** Every listener so far was distilled from full rankings: gpt-oss
orders all ~20 words for every clue, and the turn model has the guesser keep
picking until the number or a wrong word. So after its real targets, a clue
for 4 still buys guesses that are close to random, and with a third of the
board ours that is often worth it. That is how the spymaster came to give
"obscure 4" for Alien, Triangle, Spot, Sound. A human stops when the clue
points at nothing else, and those extra guesses do not exist.

**The listener.** `cache/listener_gbt_stop.txt`
(scripts/pipeline/train_stop_listener.py), trained on graded relatedness
labels (codenames/relatedness.py): the words gpt-oss says a clue points at,
asked without the number and without ranking, ordered by its stored ranking.
At pick j the guesser picks among the remaining words and STOP, by softmax
of the booster's scores, and the turn ends at STOP, at a word that is not
ours, or after the number.

**The number says what the clue points at.** This listener never sees the
number (the labels had none), so to it the number is only a cap, and a larger
one is nearly free: it only adds picks the guesser had already chosen to
make. Choosing the number by P(win) gave 4 on 26 of 30 boards, and a human
reading "4" chases 4 words. So the number is set first, as the own words the
clue points at: the expected count of own words the guesser picks before it
stops, a wrong pick not ending the count (a miss means the guesser misread
the clue, not that the clue points at fewer words), rounded, between 1 and
the cap. `points_at` simulates it (`SAMPLES` runs, fixed seed).

**The search.** win_prob_listener's frozen search with the assoc profile
booster picks the shortlist (its `shortlist` clues). Every shortlisted clue
is then valued under the STOP listener at its own number: P(win) after the
turn, with V at each ending. STOP ends the turn like a neutral word, with
nothing lost. The clue with the highest value is played. Pick j's scores do
not depend on the number or on which words were picked, so a clue needs
DEPTH booster rows per word, and the turn's value is an exact recursion over
which own words have been found.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from codenames.board import Board, OpponentBoardView, clue_number_cap
from codenames.game import Role
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.win_prob_listener import WinProbListenerSpymaster

PROFILE_MODEL = "listener_gbt_assoc_profile.txt"
STOP_MODEL = "listener_gbt_stop.txt"
DROP = ["k"]
CLUE_LEVEL = ["n_candidates", "peak_z", "lead_margin", "orth_contains", "b_assoc_distinct", "b_assoc_overlap",
              "b_assoc_first", "b_rarity", "b_senses"]
EXTRA = ["x", "is_stop"]
DEPTH = 9                       # picks simulated for `points_at`: the training depth
SAMPLES = 1000                  # sd ~0.07 on the count; ~0.6 s per 200 clues


def turn_values(S: np.ndarray, n_own: int, roles: list[Role], V: np.ndarray, K: int) -> np.ndarray:
    """P(win) after the turn for numbers 1..K. S is (K, n_words + 1): pick
    j's scores for the words (own first, then `roles`' order) and STOP last.
    Exact: a recursion over the set of own words found, each pick by softmax
    over the words left and STOP."""
    n = S.shape[1] - 1
    b = sum(r == Role.OPPONENT for r in roles)
    lam = np.exp(S - S.max(axis=1, keepdims=True))

    def after(left: int, b_left: int) -> float:
        """Our P(win) when our turn ends with `left` own words and the
        opponent's `b_left` words to go (the opponent to move)."""
        return 1.0 if left == 0 else 1.0 - V[b_left, left]

    # Value of a turn that ends on each bad word, with `left` own words to go.
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
            v = (w_bad @ end_bad[left] + w_stop * after(left, b)) / z
            for i, w in zip(own_left, w_own):
                v += w / z * value(found | {i})
            memo[found] = v
            return v

        out[k - 1] = value(frozenset())
    return out


def points_at(S: np.ndarray, n_own: int, samples: int = SAMPLES, seed: int = 0) -> float:
    """Expected own words picked before STOP, a wrong pick not ending the
    turn. S is (depth, n_words + 1) as in `turn_values`; the guesser picks
    at most `depth` times. Simulated with Gumbel noise."""
    depth, n = S.shape[0], S.shape[1] - 1
    rng = np.random.default_rng(seed)
    picked = np.zeros((samples, n), dtype=bool)
    going = np.ones(samples, dtype=bool)
    count = np.zeros(samples)
    rows = np.arange(samples)
    for j in range(depth):
        g = S[j] + rng.gumbel(size=(samples, n + 1))
        g[:, :n][picked] = -np.inf
        choice = g.argmax(axis=1)
        going &= choice < n
        if not going.any():
            break
        picked[rows[going], choice[going]] = True
        count += going & (choice < n_own)
    return float(count.mean())


class StopListenerSpymaster(WinProbListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None,
                 stop_model_path: Path | None = None, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir, model_path=Path(model_path or cache_dir / PROFILE_MODEL), **kwargs)
        if self.turn_model != "frozen" or self.calibration is not None or self.outside_n:
            raise ValueError("stop_listener's own turn model replaces the frozen one; no calibration or outside words")
        self.stop_booster = lgb.Booster(model_file=str(stop_model_path or cache_dir / STOP_MODEL))
        names = self.bundle.booster.feature_name()
        self._keep_cols = [i for i, nm in enumerate(names) if nm not in DROP]
        kept = [names[i] for i in self._keep_cols]
        if self.stop_booster.feature_name() != kept + EXTRA:
            raise ValueError("the STOP booster must read the listener's columns without k, plus x and is_stop")
        self._clue_pos = [kept.index(c) for c in CLUE_LEVEL]

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        p = {**params, "model_path": params.get("model_path") or cache_dir / PROFILE_MODEL}
        return [*super().model_files(p), cache_dir / "assoc_sims.npz", cache_dir / "assoc_profile.npz",
                Path(params.get("stop_model_path") or cache_dir / STOP_MODEL)]

    def stop_scores(self, clues: list[str], words: list[str], K: int, sims) -> dict[int, np.ndarray]:
        """{index into `clues`: (K, len(words) + 1) scores}, pick j's row for
        the words then STOP, for every clue with features, in one booster call."""
        blocks, index = [], []
        for t, c in enumerate(clues):
            f = self._listener_features(c, words, K, sims)
            if f is None:
                continue
            x = np.asarray(f, dtype=np.float64)[:, self._keep_cols]
            for j in range(1, K + 1):
                stop = np.full((1, x.shape[1] + 2), np.nan)
                stop[0, self._clue_pos] = x[0, self._clue_pos]
                stop[0, -2:] = [j, 1.0]
                blocks.append(np.vstack([np.hstack([x, np.tile([j, 0.0], (len(x), 1))]), stop]))
            index.append(t)
        if not blocks:
            return {}
        flat = np.asarray(self.stop_booster.predict(np.vstack(blocks), raw_score=True), dtype=np.float64)
        per = K * (len(words) + 1)
        return {t: flat[i * per:(i + 1) * per].reshape(K, len(words) + 1) for i, t in enumerate(index)}

    @staticmethod
    def _board(board):
        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        bad = {r: board.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        return own, words, roles

    def _score_all_clues(self, board: Board | OpponentBoardView, sims):
        best_n, scores, margin = super()._score_all_clues(board, sims)
        own, words, roles = self._board(board)
        finite = np.flatnonzero(np.isfinite(scores))
        if not own or finite.size == 0:
            return best_n, scores, margin
        cap = clue_number_cap(len(own), self.max_number)
        S = self.stop_scores([sims.clue_words[i] for i in finite], words, max(DEPTH, cap), sims)
        scores, best_n = np.full_like(scores, -np.inf), best_n.copy()
        for t, s in S.items():
            k = self.number(s, len(own), cap)
            scores[finite[t]] = turn_values(s[:k], len(own), roles, self.value.V, k)[-1]
            best_n[finite[t]] = k
        return best_n, scores, margin

    @staticmethod
    def number(S: np.ndarray, n_own: int, cap: int) -> int:
        """The number announced: the own words the clue points at, rounded, in [1, cap]."""
        return int(min(max(np.floor(points_at(S, n_own) + 0.5), 1), cap))

    def clue_value(self, board, clue: str, number: int, sims) -> float | None:
        """P(win) after this clue's turn (the play server's explanation)."""
        own, words, roles = self._board(board)
        s = self.stop_scores([clue], words, number, sims).get(0)
        return None if s is None or not own else float(turn_values(s, len(own), roles, self.value.V, number)[-1])

    def clue_points_at(self, board, clue: str, sims) -> float | None:
        """The expected own words this clue points at, before rounding."""
        own, words, _ = self._board(board)
        s = self.stop_scores([clue], words, DEPTH, sims).get(0)
        return None if s is None or not own else points_at(s, len(own))

    def listen(self, board, clue: str, sims) -> dict | None:
        """The first pick under the STOP listener, with STOP shown as the pass."""
        own, words, roles = self._board(board)
        s = self.stop_scores([clue], words, 1, sims).get(0)
        if s is None:
            return None
        return {"words": words, "roles": [Role.OWN] * len(own) + roles, "scores": s[0, :-1],
                "outside": float(s[0, -1])}
