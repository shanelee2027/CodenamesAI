"""Wraps a base guesser and stops guessing once the next-best candidate's
score falls below a threshold (SCOPE.md §3: "one with a confidence
threshold that stops early"). The only guesser type that actually
truncates its own ranking rather than just reordering the full candidate
list -- see the early-stop note in Guesser.rank_candidates()."""

from __future__ import annotations

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class ConfidenceThresholdGuesser(Guesser):
    def __init__(self, base: Guesser, threshold: float):
        self.base = base
        self.threshold = threshold

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        return self.base.score_candidates(clue, candidate_words, sims)

    def rank_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> list[str]:
        ranked = self.base.rank_candidates(clue, candidate_words, sims)
        scores = self.score_candidates(clue, candidate_words, sims)
        stopped = []
        for w in ranked:
            if scores[w] < self.threshold:
                break
            stopped.append(w)
        return stopped

    def __repr__(self) -> str:
        return f"ConfidenceThresholdGuesser(base={self.base!r}, threshold={self.threshold!r})"
