"""GPU-batched version of codenames.clue_search.mean_from_columns, batched
across many independent samples at once -- for scripts/generate_training_data.py's
clue-sampling step, which measured as the dominant per-example cost during
training-data generation (~3ms/example, ~96% of the total -- see
docs/log.md's GPU-data-generation entries).

Unlike codenames/gpu_arena.py's batching (bounded to however many games run
in parallel, naturally small), training examples are fully independent and
generated in the millions, so the natural batch size is much larger --
large enough that a naive single "gather everything, compute everything"
call runs out of GPU memory (measured: OOM around batch=2000, and even
batch=512 was already *slower* per-sample than batch=32, from memory
pressure alone). `batched_mean_similarity` chunks internally to stay in a
safe range instead of leaving that tuning to every caller.
"""

from __future__ import annotations

import numpy as np
import torch

from codenames.gpu_features import tensor_on_gpu
from codenames.similarity import SimilarityTensor

# Measured sweet spot: ~0.15-0.2ms/sample at batch 32-128 (vs. ~5ms/sample
# CPU, ~25-30x), degrading sharply above ~a few hundred as intermediate
# tensors (batch x n_clues x max_words x n_spaces, plus several
# same-shaped derived tensors) grow past what fits comfortably in VRAM.
DEFAULT_CHUNK_SIZE = 64


def _mean_similarity_chunk(tensor_gpu: torch.Tensor, sims: SimilarityTensor, word_lists: list[list[str]], device: torch.device) -> torch.Tensor:
    """(len(word_lists), n_clues), matching codenames.clue_search.
    mean_from_columns's per-sample output exactly -- NaN-aware mean across
    every (word, space) pair in that sample's word list."""
    n = len(word_lists)
    n_clues, _, n_spaces = tensor_gpu.shape
    max_words = max(len(w) for w in word_lists)

    idx_arr = torch.zeros((n, max_words), dtype=torch.long, device=device)
    valid = torch.zeros((n, max_words), dtype=torch.bool, device=device)
    for i, words in enumerate(word_lists):
        idxs = [sims.board_index[w.lower()] for w in words]
        idx_arr[i, : len(idxs)] = torch.tensor(idxs, device=device)
        valid[i, : len(idxs)] = True

    gathered = tensor_gpu[:, idx_arr, :].permute(1, 0, 2, 3)  # (n, n_clues, max_words, n_spaces)
    flat = gathered.reshape(n, n_clues, max_words * n_spaces)
    valid_flat = valid.unsqueeze(-1).expand(-1, -1, n_spaces).reshape(n, max_words * n_spaces)
    valid_flat = valid_flat.unsqueeze(1).expand(-1, n_clues, -1)

    usable = valid_flat & ~torch.isnan(flat)
    zeroed = torch.where(usable, flat, torch.zeros_like(flat))
    counts = usable.sum(dim=-1)
    sums = zeroed.sum(dim=-1)
    return torch.where(counts > 0, sums / counts.clamp(min=1), torch.full_like(sums, float("nan")))


def batched_mean_similarity(
    sims: SimilarityTensor, word_lists: list[list[str]], device: torch.device, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> np.ndarray:
    """(len(word_lists), n_clues) float32 numpy array -- one row per
    sample, each matching what `mean_from_columns([_cached_column(sims, w)
    for w in words])` would give for that sample's word list, but computed
    for the whole batch in a handful of chunked GPU calls instead of one
    CPU call per sample."""
    if not word_lists:
        return np.empty((0, len(sims.clue_words)), dtype=np.float32)
    tensor_gpu = tensor_on_gpu(sims, device)
    chunks = []
    for start in range(0, len(word_lists), chunk_size):
        chunk = word_lists[start : start + chunk_size]
        with torch.no_grad():
            chunks.append(_mean_similarity_chunk(tensor_gpu, sims, chunk, device).cpu().numpy())
    return np.concatenate(chunks, axis=0)
