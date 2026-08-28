"""Shared helpers for the baseline codemasters (SCOPE.md §6, items 1-3).

Kept separate from base.py to avoid duplicating "find the best legal clue"
and "how many words does this clue safely cover" logic across
random_clue.py, centroid.py, and linear_scorer.py.
"""

from __future__ import annotations

import random

import numpy as np

from codenames.board import Board, Role, is_legal_clue
from codenames.similarity import SimilarityTensor

# Legality failures are rare (a candidate clue has to literally contain or
# be contained by a board word); checking this many top-scored candidates
# before falling back to a full scan keeps the common case fast without
# giving up correctness in the rare case.
_CANDIDATE_POOL = 200


def top_legal_clue(sims: SimilarityTensor, board: Board, scores: np.ndarray) -> str:
    """Highest-scoring legal clue in the vocabulary. `scores` is a
    (n_clues,) array aligned with sims.clue_words; NaN entries are treated
    as unranked and never chosen."""
    finite = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    n = len(finite)
    k = min(_CANDIDATE_POOL, n)
    top_idx = np.argpartition(-finite, k - 1)[:k]
    top_idx = top_idx[np.argsort(-finite[top_idx])]
    for idx in top_idx:
        if finite[idx] == -np.inf:
            break
        clue = sims.clue_words[idx]
        if is_legal_clue(clue, board.words):
            return clue

    # The whole candidate pool was illegal or unscored -- rare, but must
    # not silently fail. Fall back to a full scan.
    order = np.argsort(-finite)
    for idx in order:
        if finite[idx] == -np.inf:
            break
        clue = sims.clue_words[idx]
        if is_legal_clue(clue, board.words):
            return clue
    raise RuntimeError("no legal, scored clue found in the vocabulary for this board")


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
