"""Baseline 1: a uniformly random legal clue."""

from __future__ import annotations

from codenames.board import Role, is_legal_clue
from codenames.similarity import SimilarityTensor

from ._util import state_rng
from .base import MAX_CLUE_NUMBER, Spymaster, TurnContext


class RandomSpymaster(Spymaster):
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        """No real ranking exists for a uniformly random pick -- each of
        the (up to) k clues drawn here is an independent random legal
        draw, not "the k best" by any score, so `score` is always 0.0.
        The first draw exactly reproduces the single-clue draw sequence
        this class always used (same rng.choice/is_legal_clue loop, same
        rng.randint(1, max_n) call), so give_clue's behavior (k=1) is
        unchanged from before this method existed."""
        board = ctx.board
        rng = state_rng(self.seed, board)
        own_remaining = board.remaining(Role.OWN)
        max_n = max(1, min(MAX_CLUE_NUMBER, own_remaining))

        clues: list[tuple[str, int, float]] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = 1000 * max(k, 1)
        while len(clues) < k and attempts < max_attempts:
            attempts += 1
            clue = rng.choice(sims.clue_words)
            if clue in seen or not is_legal_clue(clue, board.words):
                continue
            seen.add(clue)
            number = rng.randint(1, max_n)
            clues.append((clue, number, 0.0))

        if not clues:
            raise RuntimeError("could not find a legal random clue after 1000 attempts")
        return clues
