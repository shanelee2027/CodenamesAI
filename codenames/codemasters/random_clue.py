"""Baseline 1 (SCOPE.md §6): a uniformly random legal clue."""

from __future__ import annotations

from codenames.board import Board, Role, is_legal_clue
from codenames.similarity import SimilarityTensor

from ._util import state_rng
from .base import MAX_CLUE_NUMBER, Codemaster


class RandomCodemaster(Codemaster):
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        rng = state_rng(self.seed, board)
        for _ in range(1000):
            clue = rng.choice(sims.clue_words)
            if is_legal_clue(clue, board.words):
                break
        else:
            raise RuntimeError("could not find a legal random clue after 1000 attempts")

        own_remaining = board.remaining(Role.OWN)
        max_n = max(1, min(MAX_CLUE_NUMBER, own_remaining))
        number = rng.randint(1, max_n)
        return clue, number
