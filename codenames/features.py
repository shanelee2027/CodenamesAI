"""Board + clue -> feature vector (SCOPE.md §2, §M7).

This is the only thing the model ever sees -- "the model never sees words,
only numbers" (§2) -- so a bug here is silent and poisons everything built
on top of it. That's also why this file exists before generate_training_data
or the scorer: SCOPE explicitly calls for permutation-invariance and masking
tests *before* anything is built on top of it (§5 M7).

Layout, for `n = len(sims.spaces)` spaces (currently 3 -- GloVe, Numberbatch,
Wikipedia2Vec; fastText joins once M4's remaining half exists, and nothing
here is hardcoded to a specific space count):

    [space_0's 25 role-sorted values] ... [space_{n-1}'s 25 role-sorted values]
    [25-slot validity mask]
    [own_remaining, turn_index, score_differential]

Total width = 25*n + 25 + 3. SCOPE's own worked example (~115) assumes all
4 planned spaces exist (25*4 + 25 + 3 = 128); with the 3 spaces built so far
it's 25*3 + 25 + 3 = 103. Both are "~115" in the sense SCOPE meant --
approximate, not a hard target to match exactly.

Two design decisions SCOPE leaves implicit, made explicit here because they
affect the model's input shape and can't be casually changed later:

1. **Per role-group, values are sorted independently per space** (§2 step 3
   is explicit about this), which means "slot k" can be a *different*
   underlying board word in different spaces. A slot's validity therefore
   cannot be represented per-space without either breaking the independent
   per-space sort or emitting one mask per space (4x the mask width for
   comparatively little information, since these are curated board words
   overwhelmingly present in every space -- unlike the clue vocabulary,
   where per-space coverage gaps are the norm, not the exception). Given
   that, this implementation emits **one shared, space-independent mask**:
   mask[i] = 1.0 iff role-position i corresponds to a currently-unrevealed
   word in that role, full stop -- it does not also track "this specific
   space happens to lack a vector for whichever word landed here." A board
   word missing a vector in one space still gets that space's own -1
   sentinel in that space's value slot; the mask stays 1.0 because the word
   itself is real and unrevealed. This slightly under-informs the model in
   the rare case a board word truly has no vector in some space, but keeps
   the mask meaningful and cheap rather than space-specific and 4x as wide.

2. **Sentinel is -1, not NaN**, even though the rest of the codebase (the
   similarity tensor itself) uses NaN for "no vector" specifically to avoid
   confusing "missing" with "confirmed zero/unrelated" (see similarity.py).
   That reasoning doesn't carry over here: NaN can't be fed into an MLP
   (it propagates and corrupts every downstream computation), so a *model
   input* needs a fixed real-valued placeholder no matter what -- SCOPE's
   own spec pairs the -1 sentinel with an explicit validity mask for
   exactly this reason. The two conventions serve different layers
   (on-disk analysis data vs. a tensor about to hit a forward pass) and
   are not in tension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codenames.board import BOARD_SIZE, ROLE_COUNTS, Board, Role
from codenames.similarity import SimilarityTensor

ROLE_ORDER: list[Role] = [Role.OWN, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN]
SENTINEL = -1.0
N_SCALARS = 3
SCALAR_NAMES = ["own_remaining", "turn_index", "score_differential"]

# Fixed offset of each role's block within the 25-slot per-space/mask arrays.
ROLE_SLOT_RANGES: dict[Role, tuple[int, int]] = {}
_offset = 0
for _role in ROLE_ORDER:
    _count = ROLE_COUNTS[_role]
    ROLE_SLOT_RANGES[_role] = (_offset, _offset + _count)
    _offset += _count
assert _offset == BOARD_SIZE


def feature_dim(n_spaces: int) -> int:
    return BOARD_SIZE * n_spaces + BOARD_SIZE + N_SCALARS


@dataclass(frozen=True)
class FeatureLayout:
    """Named slices into a feature vector built with `n_spaces` spaces --
    lets §6 baseline 4 (a linear model over this vector) report which
    spaces/positions carry weight, without hunting for magic offsets."""

    spaces: list[str]

    def space_slice(self, space: str) -> slice:
        i = self.spaces.index(space)
        return slice(i * BOARD_SIZE, (i + 1) * BOARD_SIZE)

    def mask_slice(self) -> slice:
        n = len(self.spaces)
        return slice(n * BOARD_SIZE, n * BOARD_SIZE + BOARD_SIZE)

    def scalar_slice(self) -> slice:
        n = len(self.spaces)
        start = n * BOARD_SIZE + BOARD_SIZE
        return slice(start, start + N_SCALARS)

    @property
    def size(self) -> int:
        return feature_dim(len(self.spaces))


def _sorted_padded_values(sims: SimilarityTensor, clue: str, words: list[str], space: str, pad_to: int) -> np.ndarray:
    out = np.full(pad_to, SENTINEL, dtype=np.float32)
    if not words:
        return out
    raw = sims.similarities_for_board(clue, words, space=space)
    valid = raw[~np.isnan(raw)]
    valid_sorted = np.sort(valid)[::-1]
    out[: len(valid_sorted)] = valid_sorted
    return out


def _role_mask(remaining: int, count: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.float32)
    out[:remaining] = 1.0
    return out


def build_features(board: Board, clue: str, sims: SimilarityTensor, turn_index: int) -> np.ndarray:
    """Feature vector for this (board state, candidate clue) pair. See the
    module docstring for the exact layout."""
    role_words = {role: board.words_by_role(role, unrevealed_only=True) for role in ROLE_ORDER}

    space_blocks = []
    for space in sims.spaces:
        block = np.concatenate(
            [
                _sorted_padded_values(sims, clue, role_words[role], space, ROLE_COUNTS[role])
                for role in ROLE_ORDER
            ]
        )
        space_blocks.append(block)

    mask = np.concatenate([_role_mask(len(role_words[role]), ROLE_COUNTS[role]) for role in ROLE_ORDER])

    own_remaining = float(board.remaining(Role.OWN))
    own_revealed = ROLE_COUNTS[Role.OWN] - board.remaining(Role.OWN)
    opponent_revealed = ROLE_COUNTS[Role.OPPONENT] - board.remaining(Role.OPPONENT)
    score_differential = float(own_revealed - opponent_revealed)
    scalars = np.array([own_remaining, float(turn_index), score_differential], dtype=np.float32)

    return np.concatenate([*space_blocks, mask, scalars])
