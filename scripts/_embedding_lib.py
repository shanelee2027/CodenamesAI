"""Shared helpers for scripts/build_similarity_tensor.py and
scripts/extend_similarity_tensor.py.

Factored out once both scripts needed the same filtered-loading,
multi-word mean-pooling, and NaN-aware batched-similarity logic -- this is
duplication that showed up from real, concrete reuse across the two
scripts, not speculative abstraction.
"""

from __future__ import annotations

import bz2
import gzip
import re
from pathlib import Path

import numpy as np
import torch

ALPHABETIC = re.compile(r"^[a-z]+$")

# Per-space config: how to open the raw file, whether it has a header line
# to skip, which token prefixes to exclude (Wikipedia2Vec mixes in
# "ENTITY/Title_Case" vectors alongside plain words), and how multi-word
# board entries are natively represented in that space's vocabulary (used
# as a first-choice lookup before falling back to mean-pooling).
SPACE_CONFIGS = {
    "glove": dict(
        default_source=Path("data/embeddings/glove/glove.6B.300d.txt"),
        opener=lambda p: open(p, encoding="utf-8"),
        skip_prefixes=(),
        multiword_join=None,
        has_header=False,
    ),
    "numberbatch": dict(
        default_source=Path("data/embeddings/raw/numberbatch-en-19.08.txt.gz"),
        opener=lambda p: gzip.open(p, "rt", encoding="utf-8"),
        skip_prefixes=(),
        multiword_join="_",
        has_header=True,
    ),
    "wikipedia2vec": dict(
        default_source=Path("data/embeddings/raw/enwiki_20180420_300d.txt.bz2"),
        opener=lambda p: bz2.open(p, "rt", encoding="utf-8"),
        skip_prefixes=("ENTITY/",),
        multiword_join=None,
        has_header=True,
    ),
}


def split_words(word: str) -> list[str]:
    return re.split(r"[\s-]+", word.lower())


def ranked_alphabetic_words(path: Path, opener, skip_prefixes: tuple[str, ...], has_header: bool, limit: int | None) -> list[str]:
    """Token-only scan (no float parsing) in file order, which is
    frequency-descending for GloVe and Wikipedia2Vec (verified empirically
    -- both start with "the"/"of"/"in"-type function words). Stops as soon
    as `limit` alphabetic tokens are collected, which for a frequency-
    ordered file means reading only a prefix of it, not the whole thing
    (e.g. Wikipedia2Vec: the first 250k qualifying tokens appear within the
    first 540k of 4.53M lines).
    """
    words: list[str] = []
    with opener(path) as f:
        if has_header:
            next(f)
        for line in f:
            tok, _, _ = line.partition(" ")
            if skip_prefixes and tok.startswith(skip_prefixes):
                continue
            if ALPHABETIC.match(tok):
                words.append(tok)
                if limit is not None and len(words) >= limit:
                    break
    return words


def all_alphabetic_words(path: Path, opener, skip_prefixes: tuple[str, ...], has_header: bool) -> set[str]:
    """Like ranked_alphabetic_words with no limit, for a space with no
    reliable frequency ordering (Numberbatch's file order is arbitrary --
    verified empirically, its first entries are junk tokens like "##")."""
    return set(ranked_alphabetic_words(path, opener, skip_prefixes, has_header, limit=None))


def load_filtered_vectors(path: Path, opener, wanted: set[str], skip_prefixes: tuple[str, ...], has_header: bool) -> dict[str, np.ndarray]:
    """Full scan, but only float-parses lines whose token is in `wanted`
    -- much cheaper than parsing every vector when `wanted` is a small
    fraction of the file (e.g. Wikipedia2Vec's 4.53M lines)."""
    vectors: dict[str, np.ndarray] = {}
    with opener(path) as f:
        if has_header:
            next(f)
        for line in f:
            tok, _, rest = line.partition(" ")
            if skip_prefixes and tok.startswith(skip_prefixes):
                continue
            if tok in wanted:
                vectors[tok] = np.fromstring(rest, sep=" ", dtype=np.float32)
    return vectors


def resolve_board_vector(word: str, vectors: dict[str, np.ndarray], multiword_join: str | None) -> np.ndarray | None:
    lw = word.lower()
    if lw in vectors:
        return vectors[lw]
    if multiword_join is not None:
        joined = re.sub(r"[\s-]+", multiword_join, lw)
        if joined in vectors:
            return vectors[joined]
    parts = split_words(word)
    if len(parts) > 1:
        part_vecs = [vectors[p] for p in parts if p in vectors]
        if len(part_vecs) == len(parts):
            return np.mean(part_vecs, axis=0)
    return None


def normalize_rows(vectors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = vectors.copy()
    norms = np.linalg.norm(out[valid], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out[valid] = out[valid] / norms
    return out


def compute_similarity(
    clue_vectors: np.ndarray, clue_valid: np.ndarray,
    board_vectors: np.ndarray, board_valid: np.ndarray,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    """Batched cosine similarity (inputs must already be row-normalized).
    Entries where the clue or board word is invalid (no vector in this
    space) are set to NaN, not zero -- zero would misleadingly read as
    "confirmed unrelated" rather than "no data" for that word in this
    space."""
    board_t = torch.from_numpy(board_vectors).to(device)
    n_clues, n_board = clue_vectors.shape[0], board_vectors.shape[0]
    result = np.empty((n_clues, n_board), dtype=np.float32)

    for start in range(0, n_clues, batch_size):
        end = min(start + batch_size, n_clues)
        batch = torch.from_numpy(clue_vectors[start:end]).to(device)
        sims = batch @ board_t.T
        result[start:end] = sims.cpu().numpy()

    invalid_mask = ~clue_valid[:, None] | ~board_valid[None, :]
    result[invalid_mask] = np.nan
    return result


def build_vectors_for_vocab(
    words: list[str], vectors: dict[str, np.ndarray], dim: int,
    is_board: bool, multiword_join: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up (or mean-pool, for board words) a vector for each word in
    `words`, from an already-loaded {token: vector} dict. Returns
    (matrix, valid_mask)."""
    matrix = np.zeros((len(words), dim), dtype=np.float32)
    valid = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        if is_board:
            v = resolve_board_vector(w, vectors, multiword_join)
        else:
            v = vectors.get(w.lower())
        if v is not None:
            matrix[i] = v
            valid[i] = True
    return matrix, valid
