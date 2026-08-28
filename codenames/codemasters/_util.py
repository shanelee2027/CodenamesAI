"""Shared helpers for the baseline codemasters (SCOPE.md §6, items 1-3).

Kept separate from base.py to avoid duplicating "how many words does this
clue safely cover" logic across random_clue.py, centroid.py, and
linear_scorer.py. "Find the best legal clue" lives in codenames/clue_search.py
instead, since M7's training-data generation needs that too and isn't a
codemaster.
"""

from __future__ import annotations

import random

import numpy as np

from codenames.board import Board, Role
from codenames.similarity import SimilarityTensor


def natural_number(sims: SimilarityTensor, board: Board, clue: str, max_number: int) -> int:
    """How many own-words this clue's similarity profile ranks above every
    other unrevealed word -- the standard Codenames convention that the
    number signals how many words are safely covered. Capped at
    max_number (see codemasters.base.MAX_CLUE_NUMBER)."""
    unrevealed = [w for w in board.words if not board.is_revealed(w)]
    values = sims.similarities_for_board(clue, unrevealed)  # (n, n_spaces)
    with np.errstate(invalid="ignore"):
        mean_values = np.nanmean(values, axis=1)
    order = np.argsort(-np.nan_to_num(mean_values, nan=-np.inf))
    count = 0
    for i in order:
        if board.role_of(unrevealed[i]) == Role.OWN:
            count += 1
        else:
            break
    return max(1, min(count, max_number)) if unrevealed else 1


def state_rng(seed: int | None, board: Board) -> random.Random:
    """A Random seeded deterministically from (instance seed, board seed,
    revealed-set) so results are reproducible regardless of process or call
    order -- important once the arena runs codemasters across worker
    processes (SCOPE §5 M6)."""
    key = repr((seed, board.seed, tuple(sorted(board.revealed))))
    return random.Random(key)
