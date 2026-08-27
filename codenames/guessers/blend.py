"""Blends similarity across multiple spaces via a weighted average
(SCOPE.md §3: "one or two blending several spaces"). Supports arbitrary
weight configs so the pool config file can define several blend variants
from this one class (e.g. uniform vs. one space weighted heavier)."""

from __future__ import annotations

import math

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class BlendGuesser(Guesser):
    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        scores = {}
        for w in candidate_words:
            total, weight_sum = 0.0, 0.0
            for space, weight in self.weights.items():
                v = sims.similarity(clue, w, space=space)
                if not math.isnan(v):
                    total += weight * v
                    weight_sum += weight
            # renormalize over whichever weighted spaces actually had a
            # vector for this word -- only score -inf if none of them did
            scores[w] = total / weight_sum if weight_sum > 0 else float("-inf")
        return scores

    def __repr__(self) -> str:
        return f"BlendGuesser(weights={self.weights!r})"
