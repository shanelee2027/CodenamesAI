"""win_prob_listener with a V that reads the board, not just the score
(docs/versions/board_value_listener.md).

**Why.** win_prob_listener values each way a turn can end by V(a, b), the
incumbent's win probability from the score alone. Two clues that both find
two words leave the same score but different boards: one may leave our
hardest word alone, or an own word next to the assassin. A board correction
(codenames/board_value.py), fitted on simulated games and checked on real
ones (docs/log.md, "Simulated games"), says how much that is worth.

**The after-board, approximated.** Each ending of the turn (stop after j own
words, or j own words then a neutral or an opponent word) has a probability
from the listener; which words were found is not tracked. The after-board is
taken as the likeliest one under the listener's scores for that clue:
- the j own words found are the clue's top j own words;
- a neutral or opponent ending reveals the clue's top-scored word of that role.
Its value is 1 - sigmoid(logit V(b', a') + board term), the opponent being to
move; the exact cases stay exact (our board cleared: 1; the opponent's last
word revealed: 0; the assassin: 0).

The approximation is good when the clue's top words are clearly ahead, which
is when it matters; when they are not, the endings' probabilities are spread
and so is any error.

**Parameters:** win_prob_listener's, plus
- `board_value_path`: the booster (cache/board_value_sim.txt);
- `board_counts_path`: the count table its input logit is read from
  (cache/board_value_sim_counts.npz).

On a board with a word outside the clue policy's tables the term is 0, and
this is win_prob_listener.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codenames.board_value import BoardValue
from codenames.game import Role
from codenames.sequential_listener import pair_similarity, sequential_gain_and_penalty
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.win_prob_listener import WinProbListenerSpymaster
from codenames.win_value import win_probability

BOARD_VALUE = "board_value_sim.txt"
BOARD_COUNTS = "board_value_sim_counts.npz"
POLICY_FILES = ("policy_features.npy", "policy_features_aux.npz")
ENDINGS = ("stop", "neutral", "opponent")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class _PerClue:
    """Stands in for WinValue in win_probability: W already per clue."""

    def __init__(self, after: dict[str, np.ndarray]):
        self._after = after

    def after(self, a: int, b: int) -> dict[str, np.ndarray]:
        return self._after


class BoardValueListenerSpymaster(WinProbListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, board_value_path: Path | None = None,
                 board_counts_path: Path | None = None, **kwargs):
        super().__init__(*args, cache_dir=cache_dir, **kwargs)
        self.board_value = BoardValue(Path(board_value_path) if board_value_path else Path(cache_dir) / BOARD_VALUE,
                                      Path(board_counts_path) if board_counts_path else Path(cache_dir) / BOARD_COUNTS,
                                      Path(cache_dir))
        self._board = None

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [*super().model_files(params),
                Path(params.get("board_value_path") or cache_dir / BOARD_VALUE),
                Path(params.get("board_counts_path") or cache_dir / BOARD_COUNTS),
                *(cache_dir / f for f in POLICY_FILES)]

    def _score_all_clues(self, board, sims):
        self._board = board                  # _clue_values needs the whole board, revealed words included
        try:
            return super()._score_all_clues(board, sims)
        finally:
            self._board = None

    def _clue_values(self, s_own, s_bad, roles, costs, max_k, s_out, words=None):
        s_own, s_bad = self._calibrated(np.asarray(s_own, dtype=np.float64), np.asarray(s_bad, dtype=np.float64), roles)
        if self.turn_model == "within_turn":
            if words is None:
                raise ValueError("the within-turn model needs the board words")
            sim = pair_similarity(words, self._vw)

            def gp(c):
                return sequential_gain_and_penalty(s_own, s_bad, c, max_k, sim, self.sequential, s_out)
        else:
            def gp(c):
                return self._gain_and_penalty(s_own, s_bad, c, max_k, s_out, words=words)

        n_own = s_own.shape[1]
        n_opp = sum(r == Role.OPPONENT for r in roles)
        after = self._after_boards(s_own, s_bad, roles, max_k, words)
        value = self.value if after is None else _PerClue(after)
        values = win_probability(gp, roles, max_k, n_own, n_opp, value)
        return values, self.value.one_word(n_own, n_opp)

    def _after_boards(self, s_own, s_bad, roles, max_k, words):
        """W per clue and ending, (n_clues, max_k + 1) each, or None to fall
        back to the score alone."""
        board = self._board
        if board is None or words is None:
            return None
        full = list(board.words)
        ids = self.board_value.word_ids(full)
        if ids is None:
            return None
        slot = {w: i for i, w in enumerate(full)}
        live = np.array([not board.is_revealed(w) for w in full])
        role_of = np.array([board.role_of(w).value for w in full])
        base = {r: (role_of == r.value) & live for r in (Role.OWN, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN)}

        C, n_own = s_own.shape
        a, b = n_own, int(base[Role.OPPONENT].sum())
        own_slots = np.array([slot[w] for w in words[:n_own]])
        bad_slots = np.array([slot[w] for w in words[n_own:]])
        found = own_slots[np.argsort(-s_own, axis=1)]                    # (C, n_own), likeliest first
        J = max_k + 1

        def top_of(role):
            cols = np.flatnonzero(np.array([r == role for r in roles]))
            return None if cols.size == 0 else bad_slots[cols[np.argmax(s_bad[:, cols], axis=1)]]   # (C,)

        hit = {"stop": None, "neutral": top_of(Role.NEUTRAL), "opponent": top_of(Role.OPPONENT)}
        # ours[c, j]: our words left after the clue's top j own words are found.
        ours = np.repeat(base[Role.OWN][None, None, :], C, axis=0).repeat(J, axis=1)   # (C, J, 25)
        for j in range(1, J):
            ours[np.arange(C)[:, None], np.arange(j, J)[None, :], found[:, j - 1][:, None]] = False
        flat = ours.reshape(C * J, 25)
        rows = np.arange(C * J)
        terms = {}
        for e in ENDINGS:
            if e == "opponent" and b == 1:
                continue                                                 # revealing their last word loses outright
            masks = {r: np.repeat(base[r][None, :], C * J, axis=0) for r in (Role.OPPONENT, Role.NEUTRAL)}
            role = {"neutral": Role.NEUTRAL, "opponent": Role.OPPONENT}.get(e)
            if role is not None and hit[e] is not None:
                masks[role][rows, np.repeat(hit[e], J)] = False
            terms[e] = self.board_value.term(ids, masks[Role.OPPONENT], flat, masks[Role.NEUTRAL],
                                             np.repeat(base[Role.ASSASSIN][None, :], C * J, axis=0)).reshape(C, J)

        L = np.log(np.clip(self.value.V, 1e-4, 1 - 1e-4) / np.clip(1 - self.value.V, 1e-4, 1))
        left = a - np.arange(J)
        cleared = left[None, :] <= 0
        lc = np.maximum(left, 1)
        out = {}
        for e in ("stop", "neutral"):
            out[e] = np.where(cleared, 1.0, 1.0 - _sigmoid(L[b, lc][None, :] + terms[e]))
        if b == 1:
            out["opponent"] = np.zeros((C, J))
        else:
            out["opponent"] = np.where(cleared, 1.0, 1.0 - _sigmoid(L[b - 1, lc][None, :] + terms["opponent"]))
        return out
