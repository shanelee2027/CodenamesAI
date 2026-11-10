"""One guesser per embedding space: scores candidates by
raw similarity in exactly one space, so it only "knows" what that space
knows -- a GloVe guesser has no opinion at all on a word absent from
GloVe, which is the whole point."""

from __future__ import annotations

import math

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class SingleSpaceGuesser(Guesser):
    def __init__(self, space: str, clue_stats_stem: str | None = None):
        """`clue_stats_stem` (e.g. "clue_stats_online"): score by the clue's z
        instead of raw cosine, standardized like expected_words. Wrapped in a
        noisy guesser with noise_std = sigma, this is exactly the naive
        Gaussian model's guesser (configs/guesser_gaussian.json)."""
        self.space = space
        self.clue_stats_stem = clue_stats_stem
        self._stats = None

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        if self.clue_stats_stem is not None and self._stats is None:
            from codenames.clue_stats import ClueStats
            self._stats = ClueStats.load(stem=self.clue_stats_stem)
            self._clue_index = {c: i for i, c in enumerate(self._stats.clue_words)}
            self._si = self._stats.space_index(self.space)
        scores = {}
        for w in candidate_words:
            v = sims.similarity(clue, w, space=self.space)
            if self._stats is not None and not math.isnan(v):
                ci = self._clue_index[clue.lower()]
                v = (v - self._stats.mean[ci, self._si]) / self._stats.std[ci, self._si]
            scores[w] = float("-inf") if math.isnan(v) else float(v)
        return scores

    def __repr__(self) -> str:
        return f"SingleSpaceGuesser(space={self.space!r}, clue_stats_stem={self.clue_stats_stem!r})"
