"""Aggregates each space's RANK of the candidates rather than raw scores
(SCOPE.md §3: "one rank-based rather than score-based"). Structurally
different from BlendGuesser: converting to rank first normalizes away
each space's own similarity scale/distribution, so a space that happens
to produce generally higher or lower magnitudes doesn't dominate a
weighted average just because of its numeric range, only because of
where it actually places a word relative to the others."""

from __future__ import annotations

import math

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class RankBasedGuesser(Guesser):
    def __init__(self, spaces: list[str]):
        self.spaces = spaces

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        ranks: dict[str, list[int]] = {w: [] for w in candidate_words}
        for space in self.spaces:
            valued = [(w, sims.similarity(clue, w, space=space)) for w in candidate_words]
            valued = [(w, v) for w, v in valued if not math.isnan(v)]
            valued.sort(key=lambda p: -p[1])
            for rank, (w, _) in enumerate(valued, start=1):
                ranks[w].append(rank)

        scores = {}
        for w in candidate_words:
            if ranks[w]:
                mean_rank = sum(ranks[w]) / len(ranks[w])
                scores[w] = -mean_rank  # lower average rank (better) -> higher score
            else:
                scores[w] = float("-inf")
        return scores

    def __repr__(self) -> str:
        return f"RankBasedGuesser(spaces={self.spaces!r})"
