"""Baseline 2: clue nearest the "centroid" of a random
own-word subset.

There's no raw embedding vector available to average -- only the
precomputed similarity tensor (the build-time/train-time split means
embedding models are never loaded again once the similarity tensor is built). Standard proxy: a
candidate clue's mean cosine similarity to a set of points approximates its
similarity to their mean. So "nearest the centroid of the subset" is
computed here as the candidate clue with the highest mean similarity to the
subset words, averaged flat across both the subset words and the available
spaces (not a nested mean-of-means -- that would let a word entirely
missing from one space count for less than a word present everywhere, which
is not what we want here).
"""

from __future__ import annotations

from codenames.board import Board, Role
from codenames.clue_search import mean_similarity_to_words, top_k_legal_clues
from codenames.similarity import SimilarityTensor

from ._util import natural_number, state_rng
from .base import MAX_CLUE_NUMBER, Spymaster, TurnContext


class CentroidSpymaster(Spymaster):
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def _score_all_clues(self, board: Board, sims: SimilarityTensor):
        rng = state_rng(self.seed, board)
        own_unrevealed = board.words_by_role(Role.OWN, unrevealed_only=True)
        subset_size = rng.randint(1, max(1, min(MAX_CLUE_NUMBER, len(own_unrevealed))))
        subset = rng.sample(own_unrevealed, k=subset_size)
        return mean_similarity_to_words(sims, subset)

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        """Uses the same (deterministic, state_rng-seeded) own-word subset
        for every clue in this call -- give_clue() (k=1) and a k>1 request
        against the same board state agree on which subset was used."""
        board = ctx.board
        scores = self._score_all_clues(board, sims)
        clues = top_k_legal_clues(sims, board, scores, k)
        return [(clue, natural_number(sims, board, clue, MAX_CLUE_NUMBER), float(scores[sims.clue_index[clue.lower()]])) for clue in clues]
