"""Memory-mapped similarity tensor loader (SCOPE.md §M2).

The tensor is built by scripts/build_similarity_tensor.py and lives in
cache/ (gitignored -- regenerate it locally, it's not checked in). Shape is
(n_clues, n_board_words, n_spaces) fp16, matching SCOPE.md §2's exact
axis order so M4 can append additional space-slices without changing this
loader's interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"


@dataclass
class SimilarityTensor:
    tensor: np.memmap  # (n_clues, n_board_words, n_spaces), fp16
    clue_words: list[str]
    board_words: list[str]
    spaces: list[str]
    clue_index: dict[str, int]
    board_index: dict[str, int]

    @classmethod
    def load(cls, cache_dir: Path = DEFAULT_CACHE_DIR) -> "SimilarityTensor":
        meta = json.loads((cache_dir / "similarity_meta.json").read_text())
        tensor = np.load(cache_dir / "similarity_tensor.npy", mmap_mode="r")
        clue_words = json.loads((cache_dir / "clue_vocab.json").read_text())
        board_words = json.loads((cache_dir / "board_vocab.json").read_text())

        if tensor.shape != (len(clue_words), len(board_words), len(meta["spaces"])):
            raise ValueError(
                f"tensor shape {tensor.shape} doesn't match vocab/space sizes "
                f"({len(clue_words)}, {len(board_words)}, {len(meta['spaces'])}) "
                "-- cache is stale or corrupt, rebuild it"
            )

        return cls(
            tensor=tensor,
            clue_words=clue_words,
            board_words=board_words,
            spaces=meta["spaces"],
            clue_index={w: i for i, w in enumerate(clue_words)},
            board_index={w.lower(): i for i, w in enumerate(board_words)},
        )

    def _space_index(self, space: str | None) -> int | slice:
        if space is None:
            return slice(None)
        return self.spaces.index(space)

    def similarity(self, clue: str, board_word: str, space: str | None = None) -> np.ndarray | float:
        """Similarity of one clue to one board word. Returns a per-space
        vector if `space` is None, else a single float."""
        ci = self.clue_index.get(clue.lower())
        if ci is None:
            raise KeyError(f"{clue!r} not in clue vocabulary")
        bi = self.board_index.get(board_word.lower())
        if bi is None:
            raise KeyError(f"{board_word!r} not in board vocabulary")
        si = self._space_index(space)
        value = self.tensor[ci, bi, si]
        return float(value) if space is not None else np.asarray(value, dtype=np.float32)

    def similarities_for_board(self, clue: str, board_words: list[str], space: str | None = None) -> np.ndarray:
        """Similarity of one clue to a list of board words, e.g. the 25
        words on a specific board. Shape (len(board_words), n_spaces), or
        (len(board_words),) if `space` is given."""
        ci = self.clue_index.get(clue.lower())
        if ci is None:
            raise KeyError(f"{clue!r} not in clue vocabulary")
        idxs = []
        for w in board_words:
            bi = self.board_index.get(w.lower())
            if bi is None:
                raise KeyError(f"{w!r} not in board vocabulary")
            idxs.append(bi)
        si = self._space_index(space)
        return np.asarray(self.tensor[ci, idxs, si], dtype=np.float32)

    def top_clues(self, board_word: str, k: int = 20, space: str | None = None) -> list[tuple[str, float]]:
        """Top-k clue-vocabulary words by similarity to a single board word,
        highest first. Used by scripts/sanity_check_sims.py."""
        bi = self.board_index.get(board_word.lower())
        if bi is None:
            raise KeyError(f"{board_word!r} not in board vocabulary")
        si = self.spaces.index(space) if space is not None else 0
        column = np.asarray(self.tensor[:, bi, si], dtype=np.float32)
        top_idx = np.argpartition(-column, k)[:k]
        top_idx = top_idx[np.argsort(-column[top_idx])]
        return [(self.clue_words[i], float(column[i])) for i in top_idx]
