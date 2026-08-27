"""Wraps a base guesser and adds Gaussian noise to its scores, modeling
human inconsistency (SCOPE.md §3: "one with Gaussian noise on
similarities"). Deliberately just one guesser among several structurally
different ones -- SCOPE.md §3 warns explicitly that a pool built as one
base guesser plus several noise levels would defeat the project's own
goal (a knowledge-blind scorer can't learn to trust rare/niche clues if
every guesser shares GloVe's blind spots)."""

from __future__ import annotations

import numpy as np

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class NoisyGuesser(Guesser):
    def __init__(self, base: Guesser, noise_std: float, seed: int | None = None):
        self.base = base
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        base_scores = self.base.score_candidates(clue, candidate_words, sims)
        noisy = {}
        for w, s in base_scores.items():
            # noise on "I have no idea" should still mean no idea, not an
            # occasional lucky guess at a word this guesser has no vector for
            noisy[w] = s if s == float("-inf") else s + float(self.rng.normal(0.0, self.noise_std))
        return noisy

    def __repr__(self) -> str:
        return f"NoisyGuesser(base={self.base!r}, noise_std={self.noise_std!r})"
