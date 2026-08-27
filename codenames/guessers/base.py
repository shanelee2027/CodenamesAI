"""Guesser pool base interface (SCOPE.md §M5/§3).

A guesser only ever sees the clue and the *unrevealed* words currently on
the table -- never roles. It's the mechanism that turns "is this a good
clue" into something simulable: a guesser scores/ranks candidate words,
and whichever ones it would pick determines the outcome.

Two methods, not one, because a single "return the sorted word list"
interface can't support the pool SCOPE.md §3 actually asks for:
NoisyGuesser needs the underlying numeric scores to perturb, and
ConfidenceThresholdGuesser needs to voluntarily return *fewer* than all
candidates. `score_candidates()` is what makes each guesser type
different; `rank_candidates()` has a sensible default (sort by score) and
is only overridden by guessers that truncate their own ranking.

Diversity in this pool must come from differences in what each guesser
knows or how it decides, not from noise sprinkled on an otherwise-
identical base (SCOPE.md §3's own explicit warning) -- exactly one
guesser type here uses noise (NoisyGuesser), and it wraps a genuinely
different base rather than being the pool's only source of diversity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from codenames.similarity import SimilarityTensor


class Guesser(ABC):
    @abstractmethod
    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        """Higher score = more likely to guess. Not required to be
        bounded or a probability. A candidate this guesser's knowledge
        source has no vector for scores -inf, not 0 -- 0 would
        misleadingly compete with a real low-but-nonzero similarity."""

    def rank_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> list[str]:
        """Candidates in the order this guesser would try them, most
        likely first. May return fewer than all candidates if this
        guesser would voluntarily stop early (see
        ConfidenceThresholdGuesser) -- the caller combines this with the
        number+1 attempt cap and turn-ending-on-a-miss rule (both handled
        by the game loop in M6, not here) to determine what's actually
        played."""
        scores = self.score_candidates(clue, candidate_words, sims)
        return sorted(candidate_words, key=lambda w: -scores[w])
