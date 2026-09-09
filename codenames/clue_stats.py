"""Cached per-clue similarity statistics, over all 400 board words.

Built once by `scripts/data/build_clue_stats.py` into
`cache/clue_stats.npz` + `cache/clue_stats_meta.json`, and loaded here.
`codenames/spymasters/expected_words.py` needs "how similar is this clue to
a board word, relative to how that clue behaves in general" -- a z-score
-- which needs the clue's own mean and standard deviation of similarity
across the full board vocabulary. Those two numbers are cheap to store
(`(n_clues, n_spaces)` each) and expensive to *not* cache: `codenames/arena.py`
spawns a worker process per game and constructs a fresh spymaster in each
one (see `docs/design-decisions.md`'s memory design note), so recomputing
mean/std from the full similarity tensor inside every worker would
multiply both the one-time cost and the resident memory by the worker
count, for a value that is identical across every worker and every game.

**Deliberately not cached: the full z-tensor.** A z-scored version of the
whole similarity tensor would be `(n_clues, n_board_words, n_spaces)`
float32 -- the same 400-board-word axis as `similarity_tensor.npy`, but at
double the per-entry width (float32 vs. fp16) with none of the sharing
benefit, since it is 100% derived from data the mmapped tensor already
holds: `(sim - mean) / std` is one subtract and one divide away from
`cache/similarity_tensor.npy`, needing no extra information beyond the
2.7MB `mean`/`std` arrays this module actually stores. Materializing it
would be a ~267MB-scale duplicate purely to save a subtract-and-divide,
which contradicts the exact memory discipline
`docs/design-decisions.md`'s memory note argues for (mmap once, share
across workers, never re-derive a private full-size copy per process).
`z_for_board` computes the z-score on demand for only the board words a
given turn actually needs (at most 25), the same "read only the columns
you need off the mmap" discipline `codenames/spymasters/linear_scorer.py`
and `codenames/clue_search.py` already follow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codenames.rollouts import clue_vocab_fingerprint
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

__all__ = ["ClueStats"]


@dataclass
class ClueStats:
    """Per-clue mean/std similarity across all 400 board words, and a
    rarity percentile, indexed by clue-vocabulary position (matching
    `SimilarityTensor.clue_words` order). General-purpose -- kept free of
    anything specific to `expected_words.py` so later models can reuse it
    from a notebook or another spymaster without pulling that one in."""

    mean: np.ndarray  # (n_clues, n_spaces) float32
    std: np.ndarray  # (n_clues, n_spaces) float32
    rarity_percentile: np.ndarray  # (n_clues,) float32
    clue_words: list[str]
    spaces: list[str]

    @classmethod
    def load(cls, cache_dir: Path = DEFAULT_CACHE_DIR) -> "ClueStats":
        """Loads the cached arrays and validates them against the live
        `SimilarityTensor` at `cache_dir`. `clue_vocab_hash` mismatch
        raises rather than silently continuing: clue indices are
        meaningless against a different vocabulary (the same reasoning
        `codenames/rollouts.py::clue_vocab_fingerprint` and
        `scripts/pipeline/featurize_rollouts.py` apply to rollout sets), and a
        silent mismatch here would corrupt every downstream score rather
        than fail loudly."""
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "clue_stats_meta.json").read_text())
        data = np.load(cache_dir / "clue_stats.npz")
        sims = SimilarityTensor.load(cache_dir=cache_dir)

        live_hash = clue_vocab_fingerprint(sims.clue_words)
        if meta["clue_vocab_hash"] != live_hash:
            raise ValueError(
                f"cache/clue_stats.npz at {cache_dir} was built against clue "
                f"vocabulary hash {meta['clue_vocab_hash']!r} ({meta['n_clues']} words), "
                f"but the similarity tensor there is {live_hash!r} "
                f"({len(sims.clue_words)} words) -- clue indices would be meaningless; "
                "rebuild with scripts/data/build_clue_stats.py"
            )

        mean = data["mean"]
        std = data["std"]
        rarity_percentile = data["rarity_percentile"]
        if mean.shape != (len(sims.clue_words), len(sims.spaces)):
            raise ValueError(
                f"clue_stats.npz mean shape {mean.shape} doesn't match the live tensor's "
                f"({len(sims.clue_words)}, {len(sims.spaces)}) -- cache is stale or corrupt"
            )

        return cls(
            mean=mean,
            std=std,
            rarity_percentile=rarity_percentile,
            clue_words=sims.clue_words,
            spaces=meta["spaces"],
        )

    def space_index(self, space: str) -> int:
        return self.spaces.index(space)

    def z_for_board(self, sims: SimilarityTensor, board_words: list[str], space: str) -> np.ndarray:
        """z-scored similarity of every clue in the vocabulary against
        `board_words`, in `space`: shape `(n_clues, len(board_words))`.

        `z[c, i] = (similarity(c, board_words[i]) - mean[c]) / std[c]`,
        i.e. how many standard deviations above or below this clue's own
        typical similarity (across all 400 board words) it sits for this
        particular board word. Reads only the requested board-word
        columns off the mmapped tensor (never the full 400), so calling
        this once per turn stays cheap regardless of how large the clue
        vocabulary is."""
        si = self.space_index(space)
        idxs = [sims.board_index[w.lower()] for w in board_words]
        sim = np.asarray(sims.tensor[:, idxs, si], dtype=np.float32)  # (n_clues, len(board_words))
        mean = self.mean[:, si][:, None]
        std = self.std[:, si][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (sim - mean) / std
