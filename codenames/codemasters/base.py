"""Codemaster interface (SCOPE.md §M6/§M8).

A codemaster picks a (clue, number) pair for the current board state. The
learned scorer (M8) and every baseline (§6) implement this same interface so
the arena can play any codemaster against any guesser without special-casing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from codenames.board import Board
from codenames.similarity import SimilarityTensor

# The eventual learned scorer (M8) outputs a distribution over k in 0..4 --
# "the number of own-words the guesser will reveal before stopping" (SCOPE
# §2). Baseline codemasters cap their chosen number at the same bound so
# every codemaster's outputs stay comparable in the arena.
MAX_CLUE_NUMBER = 4


class Codemaster(ABC):
    @abstractmethod
    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        """Return (clue, number) for the current (possibly partially
        revealed) board state. `number` is the count of own-words the
        codemaster intends the clue to cover."""
        raise NotImplementedError
