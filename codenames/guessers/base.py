"""Guesser base interface.

A guesser only ever sees the clue and the *unrevealed* words currently on
the table -- never roles. It's the mechanism that turns "is this a good
clue" into something simulable: a guesser scores/ranks candidate words,
and whichever ones it would pick determines the outcome.

Two methods, not one: `NoisyGuesser` needs the underlying numeric scores to
perturb, so `score_candidates()` is what each guesser type implements, and
`rank_candidates()` has a sensible default (sort by score) that the LLM
guessers override, since they return a ranking rather than scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from codenames.similarity import SimilarityTensor


# The token a stop-capable guesser ranks to end its turn early. Defined here,
# with the interface, so the game loop and the guessers can both import it
# without importing each other (codenames/game.py, "Stopping").
STOP = "STOP"


class Guesser(ABC):
    @abstractmethod
    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        """Higher score = more likely to guess. Not required to be
        bounded or a probability. A candidate this guesser's knowledge
        source has no vector for scores -inf, not 0 -- 0 would
        misleadingly compete with a real low-but-nonzero similarity."""

    def rank_candidates(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int | None = None,
    ) -> list[str]:
        """Candidates in the order this guesser would try them, most likely
        first. The game loop applies the attempt cap and the
        turn-ends-on-a-miss rule. `number` is the announced clue number,
        which only the LLM guessers read (it goes in their prompt)."""
        scores = self.score_candidates(clue, candidate_words, sims)
        return sorted(candidate_words, key=lambda w: -scores[w])
