"""One guesser per embedding space (SCOPE.md §3): scores candidates by
raw similarity in exactly one space, so it only "knows" what that space
knows -- a GloVe guesser has no opinion at all on a word absent from
GloVe, which is the whole point (SCOPE.md §3's Technoblade example)."""

from __future__ import annotations

import math

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class SingleSpaceGuesser(Guesser):
    def __init__(self, space: str):
        self.space = space

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        scores = {}
        for w in candidate_words:
            v = sims.similarity(clue, w, space=self.space)
            scores[w] = float("-inf") if math.isnan(v) else float(v)
        return scores

    def __repr__(self) -> str:
        return f"SingleSpaceGuesser(space={self.space!r})"
