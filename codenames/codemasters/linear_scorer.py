"""Baseline 3 (SCOPE.md §6): the 8-constant linear design -- unweighted
average across spaces per (clue, board word), then a weighted sum across
roles. Uses SCOPE's own example constants (own +1, opponent -1, neutral
-0.3, assassin -10) unweighted across spaces, same convention as
scripts/inspector.py's untuned preview. §4 calls for these constants to be
CMA-ES/grid-search tuned against the guesser pool -- that tuning needs the
arena this milestone builds, so it's deferred to a follow-up script rather
than done here; this class takes `weights` as a constructor arg specifically
so a tuning script can plug in fitted values later without touching this
file.

Deliberately does NOT cache a materialized copy of the tensor: an earlier
version cached np.nanmean(tensor, axis=2) once per instance (n_clues x
n_board_words, ~850MB as float32) to avoid rescanning per turn, but the
arena runs one instance per worker process, and the mmapped-tensor sharing
that SCOPE §7's memory design note relies on only works for the read-only
mmap itself -- a materialized derived array is a private copy per process.
At even a handful of workers that cache alone was pushing worker RSS past
9GB. Instead every call reads only the board-word columns it actually needs
(at most 24) directly off the memmap; the OS page cache still shares the
underlying pages across workers, and the per-call cost is small since only
those columns are touched.
"""

from __future__ import annotations

import numpy as np

from codenames.board import Board, Role
from codenames.clue_search import top_legal_clue
from codenames.similarity import SimilarityTensor

from ._util import natural_number
from .base import MAX_CLUE_NUMBER, Codemaster

DEFAULT_WEIGHTS: dict[Role, float] = {
    Role.OWN: 1.0,
    Role.OPPONENT: -1.0,
    Role.NEUTRAL: -0.3,
    Role.ASSASSIN: -10.0,
}


class LinearScorerCodemaster(Codemaster):
    def __init__(self, weights: dict[Role, float] | None = None):
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)

    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        total = np.zeros(len(sims.clue_words), dtype=np.float32)
        for role, weight in self.weights.items():
            words = board.words_by_role(role, unrevealed_only=True)
            if not words:
                continue
            idxs = [sims.board_index[w.lower()] for w in words]
            columns = np.asarray(sims.tensor[:, idxs, :], dtype=np.float32)  # (n_clues, n_words, n_spaces)
            with np.errstate(invalid="ignore"):
                role_mean = np.nanmean(columns, axis=(1, 2))
            total += weight * np.nan_to_num(role_mean, nan=0.0)

        clue = top_legal_clue(sims, board, total)
        number = natural_number(sims, board, clue, MAX_CLUE_NUMBER)
        return clue, number
