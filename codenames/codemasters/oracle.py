"""An oracle codemaster -- not a realistic strategy, an explicit exploration
tool (per user request). For every candidate clue, sorts the board's
*unrevealed* words by raw cosine similarity in one fixed space (no noise,
no guesser pool -- just the tensor's stored value for that space) and finds
the length of the run of own-words sitting at the very top of that ranking
before the first non-own word. Picks the clue maximizing that run length
(ties broken by the mean similarity across the run).

This assumes perfect, noise-free knowledge of exactly how one specific,
fixed listener (raw cosine similarity in one embedding space) would rank
every word -- something a real spymaster can never actually have (the
entire premise of the guesser pool and the noise wrapper elsewhere in this
project is that real listeners are uncertain and varied). It exists purely
to show an upper bound: "if you somehow had perfect knowledge of exactly
how someone would rank every word by this one space, what's the best you
could possibly do?" -- useful as a reference point against the learned
model and the noisy pool, not as something to actually play with.
"""

from __future__ import annotations

import numpy as np

from codenames.board import Board, Role
from codenames.clue_search import top_k_legal_clues, top_legal_clue
from codenames.similarity import SimilarityTensor

from .base import Codemaster


class OracleCodemaster(Codemaster):
    def __init__(self, space: str = "numberbatch"):
        self.space = space

    def _score_all_clues(self, board: Board, sims: SimilarityTensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (run_length, combined_score) per clue in sims.clue_words.
        run_length is the actual consecutive-own-word count (what gets
        reported); combined_score is what ranking should sort by -- run
        length dominates, mean similarity across the run breaks ties."""
        unrevealed = [w for w in board.words if not board.is_revealed(w)]
        idxs = [sims.board_index[w.lower()] for w in unrevealed]
        space_idx = sims.spaces.index(self.space)
        values = np.asarray(sims.tensor[:, idxs, space_idx], dtype=np.float32)  # (n_clues, n_unrevealed)
        values = np.nan_to_num(values, nan=-np.inf)

        is_own = np.array([board.role_of(w) == Role.OWN for w in unrevealed])

        order = np.argsort(-values, axis=1)
        sorted_is_own = is_own[order]
        sorted_values = np.take_along_axis(values, order, axis=1)

        # Leading run of True's per row: cumulative product is 1 while
        # every entry so far is True, and collapses to 0 at (and after)
        # the first False -- summing it counts exactly the leading run.
        run_mask = np.cumprod(sorted_is_own, axis=1, dtype=np.int32)
        run_length = run_mask.sum(axis=1)

        masked_values = np.where(run_mask.astype(bool), sorted_values, 0.0)
        mean_top = masked_values.sum(axis=1) / np.maximum(run_length, 1)

        combined = run_length.astype(np.float32) * 1000.0 + mean_top
        return run_length, combined

    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        # number = the intended word count directly (matches
        # codemasters/_util.py::natural_number's convention) -- announcing
        # n grants exactly n guesses (codenames.game.play_turn), no bonus
        # attempt. Floored at 1, same as every other codemaster here: a run length
        # of 0 (this space's single best-ranked word isn't even own) still
        # has to be announced as *something*.
        run_length, combined = self._score_all_clues(board, sims)
        clue = top_legal_clue(sims, board, combined)
        clue_idx = sims.clue_index[clue.lower()]
        return clue, max(1, int(run_length[clue_idx]))

    def top_k_clues(self, board: Board, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        run_length, combined = self._score_all_clues(board, sims)
        clues = top_k_legal_clues(sims, board, combined, k)
        return [
            (clue, max(1, int(run_length[sims.clue_index[clue.lower()]])), float(run_length[sims.clue_index[clue.lower()]]))
            for clue in clues
        ]
