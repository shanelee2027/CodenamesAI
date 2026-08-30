"""Shared "search the clue vocabulary for a good, legal clue" helpers.

Used by the baseline codemasters (§6 items 2-3, codenames/codemasters/) and
by M7's training-data generation (scripts/generate_training_data.py), which
both need to turn a (n_clues,)-shaped score array into legal clue words.
"""

from __future__ import annotations

import numpy as np
from wordfreq import zipf_frequency

from codenames.board import Board, is_legal_clue
from codenames.similarity import SimilarityTensor

# Legality failures are rare (a candidate clue has to literally contain or
# be contained by a board word); checking this many top-scored candidates
# before falling back to a full scan keeps the common case fast without
# giving up correctness in the rare case.
_CANDIDATE_POOL = 200


def clue_rarity_percentile(clue_words: list[str]) -> dict[str, float]:
    """0.0 = the most common word in the clue vocabulary, ~100.0 = the
    rarest -- lets a codemaster (or the web UI) filter out obscure clues
    like "confectionery".

    Originally derived from GloVe's own frequency-ordered file position,
    but that was a bad proxy for "a person would recognize this word" --
    proper nouns (city names especially) get mentioned constantly in the
    news/web/Wikipedia text GloVe was trained on regardless of whether an
    average speaker actually knows them, so e.g. "Stuttgart" and
    "Helsinki" both landed in the top 10% by that measure (confirmed
    empirically, not assumed -- see docs/log.md). `wordfreq.zipf_frequency`
    blends subtitle/conversational-text frequency in alongside web text
    specifically to correct for that skew (subtitle frequency is the
    standard psycholinguistic fix for "recognizable word" vs. "frequently
    printed word"), fully offline after install (bundled data, no network
    calls at runtime).

    Percentile is computed within the clue vocabulary itself (not all of
    wordfreq's English vocabulary), since that's the pool an actual filter
    choice is made over -- clue_words already skews toward moderately-
    common words by construction (build_similarity_tensor.py's top-N +
    intersection filtering), so a percentile against the full English
    lexicon would make even a fairly obscure Codenames clue look
    deceptively "common."
    """
    scores = np.array([zipf_frequency(w, "en") for w in clue_words])
    order = np.argsort(-scores)  # descending: highest zipf (most common) first
    percentile = np.empty(len(clue_words), dtype=np.float64)
    percentile[order] = np.arange(len(clue_words)) / len(clue_words) * 100.0
    return dict(zip(clue_words, percentile))


def top_k_legal_clues(sims: SimilarityTensor, board: Board, scores: np.ndarray, k: int) -> list[str]:
    """Up to k highest-scoring legal clues, highest first. `scores` is a
    (n_clues,) array aligned with sims.clue_words; NaN entries are treated
    as unranked and never chosen. Returns fewer than k if the vocabulary
    doesn't have that many legally-scored candidates."""
    finite = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    n = len(finite)
    pool_size = min(max(_CANDIDATE_POOL, k), n)
    top_idx = np.argpartition(-finite, pool_size - 1)[:pool_size]
    top_idx = top_idx[np.argsort(-finite[top_idx])]

    legal: list[str] = []
    for idx in top_idx:
        if finite[idx] == -np.inf:
            break
        clue = sims.clue_words[idx]
        if is_legal_clue(clue, board.words):
            legal.append(clue)
        if len(legal) >= k:
            return legal

    if len(legal) >= k or pool_size >= n:
        return legal

    # The candidate pool didn't have k legal clues and the full vocabulary
    # is bigger than the pool we checked -- rare, but must not silently
    # under-deliver. Fall back to a full scan.
    order = np.argsort(-finite)
    legal = []
    for idx in order:
        if finite[idx] == -np.inf:
            break
        clue = sims.clue_words[idx]
        if is_legal_clue(clue, board.words):
            legal.append(clue)
        if len(legal) >= k:
            break
    return legal


def top_legal_clue(sims: SimilarityTensor, board: Board, scores: np.ndarray) -> str:
    """Highest-scoring legal clue in the vocabulary. See top_k_legal_clues."""
    top = top_k_legal_clues(sims, board, scores, k=1)
    if not top:
        raise RuntimeError("no legal, scored clue found in the vocabulary for this board")
    return top[0]


def mean_from_columns(columns: list[np.ndarray]) -> np.ndarray:
    """Per-clue mean, flat across a list of already-fetched (n_clues,
    n_spaces) columns -- shape (n_clues,), NaN where every entry across the
    given columns is missing. Split out from mean_similarity_to_words so a
    caller doing many repeated lookups of the *same* board word (e.g. M7's
    data generation sampling one of only ~400 possible board words millions
    of times) can cache the disk read once per word instead of re-reading
    it -- see scripts/generate_training_data.py for that caching layer."""
    flat = np.stack(columns, axis=1).reshape(columns[0].shape[0], -1)
    with np.errstate(invalid="ignore"):
        valid_counts = np.sum(~np.isnan(flat), axis=1)
        sums = np.nansum(flat, axis=1)
    means = np.full(sums.shape, np.nan, dtype=np.float32)
    has_data = valid_counts > 0
    means[has_data] = sums[has_data] / valid_counts[has_data]
    return means


def mean_similarity_to_words(sims: SimilarityTensor, words: list[str]) -> np.ndarray:
    """Per-clue mean similarity, flat across the given board words and
    spaces -- shape (n_clues,). NaN where every (word, space) pair is
    missing. Doubles as a "distance to centroid" proxy: a candidate clue's
    mean cosine similarity to a set of points approximates its similarity
    to their mean, which is as close as we can get without raw embedding
    vectors (see codemasters/centroid.py's docstring for the full reasoning)."""
    cols = [np.asarray(sims.tensor[:, sims.board_index[w.lower()], :], dtype=np.float32) for w in words]
    return mean_from_columns(cols)
