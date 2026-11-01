"""Word-kind features: what sort of word a board word is, independent of the clue.

The word-difficulty test (scripts/tools/word_difficulty.py, docs/log.md
2026-11-01) found two kinds of board word the listener booster misjudges:
proper names, which it overrates (picked 0.91x as often as predicted), and
short common words with many meanings, which it underrates by up to 2.5x
(CODE, SEAL, KEY, BRIDGE): the guesser reaches for them from clues the
features see no link in. Neither is a property of a (clue, word) pair, so
no similarity feature carries it. Three columns, each the same on every row
for a word:

- `proper_share`: the share of the word's WordNet senses that are named
  instances (Homer: one of three; Australia: all; code: none). NaN if
  WordNet lacks the word.
- `hub_assoc`: the share of the clue pool whose gpt-oss association lists
  (cache/assoc_sims.npz) name the word at least once. How many unrelated
  clues lead the guesser's own model to it.
- `hub_z`: the share of the clue pool for which the word is a strong
  outlier, z >= 2 against the clue's mean and sd over the online game's 800
  (clue_stats_online), averaged over the tensor's three spaces. The
  embedding analogue of the same thing (k-occurrence "hubness").

All three are measured against fixed references, the clue pool and the
online 800, so appending board words later never moves an existing word's
value. Built by scripts/data/build_word_kind.py into cache/word_kind.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

KIND_FEATURES = ("proper_share", "hub_assoc", "hub_z")


class WordKind:
    def __init__(self, board_pos: dict[str, int], values: np.ndarray):
        self.board_pos = board_pos             # lower-cased board word -> row
        self.values = values                   # (n_board_words, len(KIND_FEATURES))

    @classmethod
    def load(cls, path: Path) -> "WordKind":
        d = np.load(path, allow_pickle=False)
        return cls({str(w).lower(): i for i, w in enumerate(d["board_words"])}, d["values"])

    def columns(self, clue: str, candidates: list[str]) -> np.ndarray:
        """(len(candidates), len(KIND_FEATURES)); the clue is not used."""
        out = np.full((len(candidates), len(KIND_FEATURES)), np.nan)
        for i, w in enumerate(candidates):
            r = self.board_pos.get(w.lower())
            if r is not None:
                out[i] = self.values[r]
        return out
