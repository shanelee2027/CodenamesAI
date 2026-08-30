"""Wraps a base guesser and adds Gaussian noise to its scores, modeling
human inconsistency (SCOPE.md §3: "one with Gaussian noise on
similarities"). Deliberately just one guesser among several structurally
different ones -- SCOPE.md §3 warns explicitly that a pool built as one
base guesser plus several noise levels would defeat the project's own
goal (a knowledge-blind scorer can't learn to trust rare/niche clues if
every guesser shares GloVe's blind spots).

Noise is a deterministic function of (seed, clue, word) -- one word's
noisy misperception of one clue is fixed, not a fresh dice roll every
time it's asked about -- rather than a draw from one continuously-
advancing RNG stream. This isn't just a style choice: codenames/guessers/
base.py's backlog/bonus-guess mechanism (see its module docstring) and
HistoryAwareGuesser's z-score baseline cache both explicitly assume
"re-scoring the same clue against the same candidates always reproduces
the same answer" -- true for every other guesser in the pool, but was
silently false here (a sequential RNG stream means the *n*-th call for a
clue depends on how many unrelated calls happened before it, so the same
clue scored twice -- once during real play, once retrospectively in
update_history's "did this backlog get satisfied" check -- could
disagree). That mismatch let an already-satisfied backlog entry look
still-owed on a later turn, spending an unearned bonus guess (see
docs/log.md's history-aware-determinism entry)."""

from __future__ import annotations

import zlib

import numpy as np

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class NoisyGuesser(Guesser):
    def __init__(self, base: Guesser, noise_std: float, seed: int | None = None):
        self.base = base
        self.noise_std = noise_std
        self.seed = 0 if seed is None else seed

    def _noise(self, clue: str, word: str) -> float:
        # zlib.crc32 (not Python's built-in hash()) specifically because
        # str hashing is randomized per-process by default -- this needs
        # to be the same value every run, not just within one process.
        entropy = [self.seed, zlib.crc32(clue.lower().encode()), zlib.crc32(word.lower().encode())]
        return float(np.random.default_rng(entropy).normal(0.0, self.noise_std))

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        base_scores = self.base.score_candidates(clue, candidate_words, sims)
        noisy = {}
        for w, s in base_scores.items():
            # noise on "I have no idea" should still mean no idea, not an
            # occasional lucky guess at a word this guesser has no vector for
            noisy[w] = s if s == float("-inf") else s + self._noise(clue, w)
        return noisy

    def __repr__(self) -> str:
        return f"NoisyGuesser(base={self.base!r}, noise_std={self.noise_std!r})"
