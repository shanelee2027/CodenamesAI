"""Codemaster interface (SCOPE.md §M6/§M8).

A codemaster picks a (clue, number) pair for the current board state. The
learned scorer (M8) and every baseline (§6) implement this same interface so
the arena can play any codemaster against any guesser without special-casing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from codenames.board import MAX_CLUE_NUMBER, Board
from codenames.similarity import SimilarityTensor

__all__ = ["Codemaster", "MAX_CLUE_NUMBER"]


class Codemaster(ABC):
    @abstractmethod
    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        """Return (clue, number) for the current (possibly partially
        revealed) board state. `number` is the count of own-words the
        codemaster intends the clue to cover."""
        raise NotImplementedError
