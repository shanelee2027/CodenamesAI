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

A third thing made explicit here: **the feature vector's per-role slot
widths are a separate concept from the real board's per-role card counts**
(`codenames.board.ROLE_COUNTS`), even though they used to be numerically
identical for every role. `FEATURE_SLOT_COUNTS` below is this file's own
copy, with OPPONENT widened to 9 (matching OWN) rather than the real
board's 8. Why: `codenames.board.OpponentBoardView` (two-team play) swaps
OWN/OPPONENT for the second team, so that team's "opponent" role is the
physical board's real 9-member OWN group -- one over the fixed 8-slot
allocation this file used to derive straight from `ROLE_COUNTS`. Widening
just this file's slot width to 9 fits that case with ordinary padding
(exactly like any role that's lost a few words to reveals), with zero
change to `Board.generate`'s real 9/8/7/1 card counts -- so no team's true
win condition changes either. See docs/log.md for the two-team-play bug
this replaced (a same-instance ROLE_COUNTS reuse that either crashed or,
before that, silently overflowed).

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

from codenames.board import ROLE_COUNTS, Board, Role
from codenames.similarity import SimilarityTensor

ROLE_ORDER: list[Role] = [Role.OWN, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN]
SENTINEL = -1.0
N_SCALARS = 3
SCALAR_NAMES = ["own_remaining", "turn_index", "score_differential"]

# Feature-vector slot widths -- deliberately a separate dict from
# codenames.board.ROLE_COUNTS (the real board's card counts). See the
# module docstring for why OPPONENT is 9 here, not 8.
FEATURE_SLOT_COUNTS: dict[Role, int] = {
    Role.OWN: 9,
    Role.OPPONENT: 9,
    Role.NEUTRAL: ROLE_COUNTS[Role.NEUTRAL],
    Role.ASSASSIN: ROLE_COUNTS[Role.ASSASSIN],
}
FEATURE_BOARD_SIZE = sum(FEATURE_SLOT_COUNTS.values())

# Fixed offset of each role's block within the FEATURE_BOARD_SIZE-slot
# per-space/mask arrays.
ROLE_SLOT_RANGES: dict[Role, tuple[int, int]] = {}
_offset = 0
for _role in ROLE_ORDER:
    _count = FEATURE_SLOT_COUNTS[_role]
    ROLE_SLOT_RANGES[_role] = (_offset, _offset + _count)
    _offset += _count
assert _offset == FEATURE_BOARD_SIZE


def feature_dim(n_spaces: int) -> int:
    return FEATURE_BOARD_SIZE * n_spaces + FEATURE_BOARD_SIZE + N_SCALARS


@dataclass(frozen=True)
class FeatureLayout:
    """Named slices into a feature vector built with `n_spaces` spaces --
    lets §6 baseline 4 (a linear model over this vector) report which
    spaces/positions carry weight, without hunting for magic offsets."""

    spaces: list[str]

    def space_slice(self, space: str) -> slice:
        i = self.spaces.index(space)
        return slice(i * FEATURE_BOARD_SIZE, (i + 1) * FEATURE_BOARD_SIZE)

    def mask_slice(self) -> slice:
        n = len(self.spaces)
        return slice(n * FEATURE_BOARD_SIZE, n * FEATURE_BOARD_SIZE + FEATURE_BOARD_SIZE)

    def scalar_slice(self) -> slice:
        n = len(self.spaces)
        start = n * FEATURE_BOARD_SIZE + FEATURE_BOARD_SIZE
        return slice(start, start + N_SCALARS)

    @property
    def size(self) -> int:
        return feature_dim(len(self.spaces))

    def describe(self, index: int) -> str:
        """Label a raw feature index for interpretability (SCOPE §9's linear
        baseline needs to report "which spaces and rank positions carry
        weight", not just raw indices)."""
        n = len(self.spaces)
        mask_start = n * FEATURE_BOARD_SIZE
        scalar_start = mask_start + FEATURE_BOARD_SIZE

        if index < mask_start:
            space_i, pos = divmod(index, FEATURE_BOARD_SIZE)
            prefix = self.spaces[space_i]
        elif index < scalar_start:
            pos = index - mask_start
            prefix = "mask"
        else:
            return f"scalar/{SCALAR_NAMES[index - scalar_start]}"

        for role, (start, end) in ROLE_SLOT_RANGES.items():
            if start <= pos < end:
                return f"{prefix}/{role.value}/rank{pos - start}"
        raise AssertionError(f"position {pos} not covered by any role range")


def _check_capacity(n_words: int, pad_to: int) -> None:
    """Every one of these padding functions assumes `n_words <= pad_to`.
    FEATURE_SLOT_COUNTS is sized so this holds for every real perspective,
    including codenames.board.OpponentBoardView's swapped one (own=8/
    opponent=9 from the second team's view) -- see the module docstring.
    This guard is defense-in-depth left over from when it wasn't sized
    that way yet: OPPONENT's slot used to just reuse ROLE_COUNTS's real-
    board count (8), and a swapped-perspective 9-word group would either
    silently overflow (a numpy assignment past an array's own length
    simply clips, or -- in the batched path -- there was no bounds check
    at all) or, once a bounds check existed, hard-crash a codemaster that
    could otherwise play fine. See docs/log.md for that history. Raising
    immediately here, with a clear message, stays the module docstring's
    own standard regardless: "a bug here is silent and poisons everything
    built on top of it."
    """
    if n_words > pad_to:
        raise ValueError(
            f"{n_words} words for a role allocated only {pad_to} slots -- see FEATURE_SLOT_COUNTS. "
            "This shouldn't happen for any role/perspective this file's slot widths were sized for "
            "(including codenames/board.py::OpponentBoardView's swapped two-team perspective) -- if it "
            "fires, something is handing build_features a board-like object with more real words in a "
            "role than any known perspective produces."
        )


def _sorted_padded_values(sims: SimilarityTensor, clue: str, words: list[str], space: str, pad_to: int) -> np.ndarray:
    _check_capacity(len(words), pad_to)
    out = np.full(pad_to, SENTINEL, dtype=np.float32)
    if not words:
        return out
    raw = sims.similarities_for_board(clue, words, space=space)
    valid = raw[~np.isnan(raw)]
    valid_sorted = np.sort(valid)[::-1]
    out[: len(valid_sorted)] = valid_sorted
    return out


def _unsorted_padded_values(sims: SimilarityTensor, clue: str, words: list[str], space: str, pad_to: int) -> np.ndarray:
    """Like _sorted_padded_values, but keeps each word's value at its own
    natural (words_by_role order) position instead of sorting descending --
    used only by build_features_unsorted (SCOPE §9's sort ablation)."""
    _check_capacity(len(words), pad_to)
    out = np.full(pad_to, SENTINEL, dtype=np.float32)
    if not words:
        return out
    raw = sims.similarities_for_board(clue, words, space=space)
    out[: len(words)] = np.where(np.isnan(raw), SENTINEL, raw)
    return out


def _role_mask(remaining: int, count: int) -> np.ndarray:
    _check_capacity(remaining, count)
    out = np.zeros(count, dtype=np.float32)
    out[:remaining] = 1.0
    return out


def _compute_mask(role_words: dict[Role, list[str]]) -> np.ndarray:
    return np.concatenate([_role_mask(len(role_words[role]), FEATURE_SLOT_COUNTS[role]) for role in ROLE_ORDER])


def _compute_scalars(board: Board, turn_index: int) -> np.ndarray:
    # True per-role totals from *this* perspective, not the ROLE_COUNTS
    # constant -- that constant is only correct for a board's own real
    # perspective (own=9, opponent=8). Under OpponentBoardView's swap,
    # "own" genuinely totals 8 and "opponent" genuinely totals 9, so
    # deriving each total from the words actually assigned that role here
    # (revealed or not) is what stays correct in both directions.
    own_total = len(board.words_by_role(Role.OWN))
    opponent_total = len(board.words_by_role(Role.OPPONENT))
    own_remaining = float(board.remaining(Role.OWN))
    own_revealed = own_total - board.remaining(Role.OWN)
    opponent_revealed = opponent_total - board.remaining(Role.OPPONENT)
    score_differential = float(own_revealed - opponent_revealed)
    return np.array([own_remaining, float(turn_index), score_differential], dtype=np.float32)


def build_features(board: Board, clue: str, sims: SimilarityTensor, turn_index: int) -> np.ndarray:
    """Feature vector for this (board state, candidate clue) pair. See the
    module docstring for the exact layout."""
    role_words = {role: board.words_by_role(role, unrevealed_only=True) for role in ROLE_ORDER}

    space_blocks = []
    for space in sims.spaces:
        block = np.concatenate(
            [
                _sorted_padded_values(sims, clue, role_words[role], space, FEATURE_SLOT_COUNTS[role])
                for role in ROLE_ORDER
            ]
        )
        space_blocks.append(block)

    mask = _compute_mask(role_words)
    scalars = _compute_scalars(board, turn_index)
    return np.concatenate([*space_blocks, mask, scalars])


def build_features_unsorted(board: Board, clue: str, sims: SimilarityTensor, turn_index: int) -> np.ndarray:
    """SCOPE §9's sort ablation: identical layout and width to
    build_features(), but each role group's values keep their natural
    (unrevealed, board-order) position instead of being sorted descending.
    Deliberately reintroduces the position-carries-no-information problem
    §2 sorts specifically to avoid, as a controlled comparison."""
    role_words = {role: board.words_by_role(role, unrevealed_only=True) for role in ROLE_ORDER}

    space_blocks = []
    for space in sims.spaces:
        block = np.concatenate(
            [
                _unsorted_padded_values(sims, clue, role_words[role], space, FEATURE_SLOT_COUNTS[role])
                for role in ROLE_ORDER
            ]
        )
        space_blocks.append(block)

    mask = _compute_mask(role_words)
    scalars = _compute_scalars(board, turn_index)
    return np.concatenate([*space_blocks, mask, scalars])


def _sorted_padded_values_batch(sims: SimilarityTensor, words: list[str], space: str, pad_to: int) -> np.ndarray:
    """Vectorized form of _sorted_padded_values: every clue in the
    vocabulary at once. Shape (n_clues, pad_to)."""
    _check_capacity(len(words), pad_to)
    n_clues = len(sims.clue_words)
    if not words:
        return np.full((n_clues, pad_to), SENTINEL, dtype=np.float32)

    idxs = [sims.board_index[w.lower()] for w in words]
    si = sims.spaces.index(space)
    raw = np.asarray(sims.tensor[:, idxs, si], dtype=np.float32)  # (n_clues, len(words))

    # NaN sorts as "biggest" in numpy, which would push missing values to
    # the *front* of a descending sort -- exactly backwards from where the
    # sentinel padding belongs. Swap to -inf first so a descending sort
    # naturally leaves real values first and missing ones trailing, then
    # replace those trailing -inf slots with the actual sentinel.
    raw = np.where(np.isnan(raw), -np.inf, raw)
    sorted_desc = np.sort(raw, axis=1)[:, ::-1]
    sorted_desc = np.where(np.isneginf(sorted_desc), SENTINEL, sorted_desc)

    if sorted_desc.shape[1] < pad_to:
        pad_width = pad_to - sorted_desc.shape[1]
        padding = np.full((n_clues, pad_width), SENTINEL, dtype=np.float32)
        sorted_desc = np.concatenate([sorted_desc, padding], axis=1)
    return sorted_desc


def build_features_batch(board: Board, sims: SimilarityTensor, turn_index: int) -> np.ndarray:
    """Feature vectors for *every* clue in the vocabulary at once, against
    one board state -- shape (n_clues, feature_dim). This is what play-time
    scoring needs: SCOPE §2 requires scoring all ~250k+ candidates via "one
    gather plus one small forward pass," which rules out calling
    build_features() once per clue in a Python loop. The mask and scalar
    blocks don't depend on the clue at all, so they're computed once and
    broadcast across every row; only the per-space value blocks vary per
    clue, and that variation comes entirely from the underlying tensor
    (already indexed by clue), not from any extra Python-level looping.
    """
    n_clues = len(sims.clue_words)
    role_words = {role: board.words_by_role(role, unrevealed_only=True) for role in ROLE_ORDER}

    space_blocks = []
    for space in sims.spaces:
        block = np.concatenate(
            [
                _sorted_padded_values_batch(sims, role_words[role], space, FEATURE_SLOT_COUNTS[role])
                for role in ROLE_ORDER
            ],
            axis=1,
        )
        space_blocks.append(block)

    mask_block = np.broadcast_to(_compute_mask(role_words), (n_clues, FEATURE_BOARD_SIZE))
    scalars_block = np.broadcast_to(_compute_scalars(board, turn_index), (n_clues, N_SCALARS))

    return np.concatenate([*space_blocks, mask_block, scalars_block], axis=1).astype(np.float32)
