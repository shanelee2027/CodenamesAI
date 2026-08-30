"""GPU-batched feature construction across *multiple boards at once* --
codenames/features.py's build_features_batch already batches across every
clue in the vocabulary for one board; this batches that across many
simultaneous boards too, entirely on GPU, for codenames/gpu_arena.py's
bulk self-play throughput.

Kept out of features.py deliberately: features.py is the pure-numpy
reference implementation, imported by scripts/generate_training_data.py
and everything else that has no reason to pull in torch. This module
depends on torch and is only ever imported by the GPU arena path.

Numerically verified against codenames.features.build_features_batch
(exact match, not just close) before this was trusted -- see
tests/test_gpu_features.py.

Why batching helps, concretely (measured, not assumed): looping this
board-by-board on GPU still costs ~25-33ms/board, barely better than
numpy's ~80ms/board CPU cost, because each board incurs 12 small GPU
kernel launches (4 roles x 3 spaces) whose overhead dominates the actual
compute. Doing the gather+sort as one tensor op across many boards at
once amortizes that overhead: ~43ms/board at batch=1 down to ~6ms/board
at batch=32 in testing, and still improving at 32 -- see docs/log.md's
GPU-arena entries for the full numbers.
"""

from __future__ import annotations

import numpy as np
import torch

from codenames.board import Board, Role
from codenames.features import FEATURE_BOARD_SIZE, FEATURE_SLOT_COUNTS, N_SCALARS, ROLE_ORDER, SENTINEL
from codenames.similarity import SimilarityTensor

# Keyed by id(sims) -- in practice a SimilarityTensor is loaded exactly
# once per process and lives for that process's whole lifetime (see
# codenames/gpu_arena.py), so id() reuse after garbage collection isn't a
# practical risk here. Not using a WeakValueDictionary/similar because
# SimilarityTensor doesn't support weak references without extra plumbing,
# and the single-long-lived-instance usage pattern doesn't need one.
_TENSOR_GPU_CACHE: dict[int, torch.Tensor] = {}


def tensor_on_gpu(sims: SimilarityTensor, device: torch.device) -> torch.Tensor:
    """The full similarity tensor, moved to `device` once and cached
    thereafter. float32, not the on-disk fp16 -- sort/compare ops on GPU
    fp16 are markedly slower and more failure-prone around NaN/-inf edge
    cases than fp32, and the tensor easily fits either way (~0.5GB at
    fp32 for the current 3-space vocabulary)."""
    key = id(sims)
    cached = _TENSOR_GPU_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = torch.from_numpy(np.asarray(sims.tensor, dtype=np.float32)).to(device)
    _TENSOR_GPU_CACHE[key] = tensor
    return tensor


def _batched_role_block(tensor_gpu: torch.Tensor, sims: SimilarityTensor, boards: list[Board], role: Role, space_idx: int, device: torch.device) -> torch.Tensor:
    """(n_boards, n_clues, pad_to), one gather + one sort call for every
    board at once."""
    pad_to = FEATURE_SLOT_COUNTS[role]
    n_boards = len(boards)
    n_clues = tensor_gpu.shape[0]
    idx_lists = [[sims.board_index[w.lower()] for w in b.words_by_role(role, unrevealed_only=True)] for b in boards]
    max_words = max((len(lst) for lst in idx_lists), default=0)
    if max_words > pad_to:
        raise ValueError(f"{max_words} words for a role allocated only {pad_to} slots -- see codenames.features.FEATURE_SLOT_COUNTS")
    if max_words == 0:
        return torch.full((n_boards, n_clues, pad_to), SENTINEL, device=device)

    idx_arr = torch.zeros((n_boards, max_words), dtype=torch.long, device=device)
    valid = torch.zeros((n_boards, max_words), dtype=torch.bool, device=device)
    for i, lst in enumerate(idx_lists):
        if lst:
            idx_arr[i, : len(lst)] = torch.tensor(lst, device=device)
            valid[i, : len(lst)] = True

    gathered = tensor_gpu[:, idx_arr, space_idx].permute(1, 0, 2)  # (n_boards, n_clues, max_words)
    invalid = ~valid.unsqueeze(1).expand(-1, n_clues, -1)
    neg_inf = torch.tensor(float("-inf"), device=device)
    raw = torch.where(torch.isnan(gathered) | invalid, neg_inf, gathered)
    sorted_desc, _ = torch.sort(raw, dim=2, descending=True)
    sorted_desc = torch.where(torch.isneginf(sorted_desc), torch.tensor(SENTINEL, device=device), sorted_desc)

    if max_words < pad_to:
        pad = torch.full((n_boards, n_clues, pad_to - max_words), SENTINEL, device=device)
        sorted_desc = torch.cat([sorted_desc, pad], dim=2)
    return sorted_desc


def build_features_batch_multi(sims: SimilarityTensor, boards: list[Board], turn_indices: list[int], device: torch.device) -> torch.Tensor:
    """Feature vectors for every clue in the vocabulary, against every one
    of `boards` at once -- shape (n_boards, n_clues, feature_dim), on
    `device`. The multi-board analog of
    codenames.features.build_features_batch (which only batches across
    clues, for one board)."""
    tensor_gpu = tensor_on_gpu(sims, device)
    n_boards = len(boards)
    n_clues = tensor_gpu.shape[0]

    space_blocks = []
    for space in sims.spaces:
        si = sims.spaces.index(space)
        block = torch.cat([_batched_role_block(tensor_gpu, sims, boards, role, si, device) for role in ROLE_ORDER], dim=2)
        space_blocks.append(block)

    masks = []
    for b in boards:
        role_words = {role: b.words_by_role(role, unrevealed_only=True) for role in ROLE_ORDER}
        parts = []
        for role in ROLE_ORDER:
            n_unrevealed = len(role_words[role])
            pad_to = FEATURE_SLOT_COUNTS[role]
            parts.append(torch.cat([torch.ones(n_unrevealed), torch.zeros(pad_to - n_unrevealed)]))
        masks.append(torch.cat(parts))
    mask = torch.stack(masks).to(device)  # (n_boards, FEATURE_BOARD_SIZE)

    scalars = []
    for b, turn_index in zip(boards, turn_indices):
        # True per-role totals from this board's own perspective, not
        # ROLE_COUNTS -- see codenames/features.py::_compute_scalars for
        # why (correct for a real board, wrong under OpponentBoardView).
        own_total = len(b.words_by_role(Role.OWN))
        opponent_total = len(b.words_by_role(Role.OPPONENT))
        own_remaining = float(b.remaining(Role.OWN))
        own_revealed = own_total - b.remaining(Role.OWN)
        opponent_revealed = opponent_total - b.remaining(Role.OPPONENT)
        scalars.append([own_remaining, float(turn_index), float(own_revealed - opponent_revealed)])
    scalars_t = torch.tensor(scalars, device=device)  # (n_boards, N_SCALARS)

    mask_b = mask.unsqueeze(1).expand(-1, n_clues, -1)
    scalars_b = scalars_t.unsqueeze(1).expand(-1, n_clues, -1)
    assert mask_b.shape[-1] == FEATURE_BOARD_SIZE and scalars_b.shape[-1] == N_SCALARS  # keep in sync with features.py's layout
    return torch.cat([*space_blocks, mask_b, scalars_b], dim=2)
