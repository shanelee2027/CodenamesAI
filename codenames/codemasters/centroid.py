"""Baseline 2 (SCOPE.md §6): clue nearest the "centroid" of a random
own-word subset.

There's no raw embedding vector available to average -- only the
precomputed similarity tensor (SCOPE §2's build-time/train-time split means
embedding models are never loaded again after M2/M4). Standard proxy: a
candidate clue's mean cosine similarity to a set of points approximates its
similarity to their mean. So "nearest the centroid of the subset" is
computed here as the candidate clue with the highest mean similarity to the
subset words, averaged flat across both the subset words and the available
spaces (not a nested mean-of-means -- that would let a word entirely
missing from one space count for less than a word present everywhere, which
is not what we want here).
"""

from __future__ import annotations

import numpy as np

from codenames.board import Board, Role
from codenames.similarity import SimilarityTensor

from ._util import natural_number, state_rng, top_legal_clue
from .base import MAX_CLUE_NUMBER, Codemaster


def _mean_similarity_to_words(sims: SimilarityTensor, words: list[str]) -> np.ndarray:
    """Per-clue mean similarity, flat across the given words and spaces --
    shape (n_clues,). NaN where every (word, space) pair is missing."""
    cols = [np.asarray(sims.tensor[:, sims.board_index[w.lower()], :], dtype=np.float32) for w in words]
    flat = np.stack(cols, axis=1).reshape(len(sims.clue_words), -1)
    with np.errstate(invalid="ignore"):
        valid_counts = np.sum(~np.isnan(flat), axis=1)
        sums = np.nansum(flat, axis=1)
    means = np.full(sums.shape, np.nan, dtype=np.float32)
    has_data = valid_counts > 0
    means[has_data] = sums[has_data] / valid_counts[has_data]
    return means


class CentroidCodemaster(Codemaster):
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        rng = state_rng(self.seed, board)
        own_unrevealed = board.words_by_role(Role.OWN, unrevealed_only=True)
        subset_size = rng.randint(1, max(1, min(MAX_CLUE_NUMBER, len(own_unrevealed))))
        subset = rng.sample(own_unrevealed, k=subset_size)

        scores = _mean_similarity_to_words(sims, subset)
        clue = top_legal_clue(sims, board, scores)
        number = natural_number(sims, board, clue, MAX_CLUE_NUMBER)
        return clue, number
