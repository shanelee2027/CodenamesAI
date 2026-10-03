"""Two listener feature blocks from gpt-oss's free associations
(docs/log.md, "Exploratory feature blocks on the assoc booster").

- **B, the clue's profile** (8 board constants, so they act only through
  interactions: a vague clue should flatten the other features' effect):
  - how vague gpt-oss finds the clue: across its association lists, the
    share of distinct words, the mean pairwise overlap of the lists, and how
    often they open with the same word;
  - its rarity percentile, concreteness, percent known and log frequency
    (Brysbaert norms) and number of WordNet senses.
- **C, reverse associations** (3 per word): from each board word's own lists
  (scripts/data/collect_associations.py --board), the share that name the
  clue, the mean reciprocal position of the clue in them, and that score's
  rank on the board.

The table (cache/assoc_profile.npz, scripts/data/build_assoc_profile.py)
holds B per clue and the lists of every board word. `columns` is the one
implementation, used both to add the columns to training positions and by
listener_features.extract in play.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B_FEATURES = ("b_assoc_distinct", "b_assoc_overlap", "b_assoc_first", "b_rarity", "b_conc", "b_known",
              "b_logfreq", "b_senses")
C_FEATURES = ("c_rev_share", "c_rev_rank", "c_rev_board_rank")
PROFILE_FEATURES = B_FEATURES + C_FEATURES


def assoc_norm(w: str) -> str:
    from codenames.listener_training import _assoc_norm

    return _assoc_norm(w)


def clue_forms(norm: str) -> set[str]:
    """The clue as it may appear in a list: plain, plural, or singular."""
    return {norm, norm + "s", norm + "es"} | ({norm[:-1]} if norm.endswith("s") else set())


def vagueness(lists: list[list[str]]) -> tuple[float, float, float]:
    """(distinct share, mean pairwise Jaccard, share opening with the modal
    first word) over a cue's lists, NaN without lists."""
    if not lists:
        return np.nan, np.nan, np.nan
    sets = [set(x) for x in lists]
    distinct = len(set().union(*sets)) / sum(len(x) for x in lists)
    pairs = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    overlap = float(np.mean(pairs)) if pairs else np.nan
    firsts = [x[0] for x in lists]
    first = max(firsts.count(f) for f in set(firsts)) / len(firsts)
    return distinct, overlap, first


def rank_desc(v: np.ndarray) -> np.ndarray:
    """1 for the board's highest value, ties averaged, NaN stays NaN."""
    from scipy.stats import rankdata

    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    if ok.any():
        out[ok] = rankdata(-v[ok], method="average")
    return out


class AssocProfile:
    def __init__(self, clue_rows: dict[str, int], b: np.ndarray, word_lists: dict[str, list[list[str]]]):
        self.clue_rows = clue_rows             # lower-cased clue -> row of b
        self.b = b                             # (n_clues, len(B_FEATURES))
        self.word_lists = word_lists           # assoc_norm(board word) -> its lists

    @classmethod
    def load(cls, path: Path) -> "AssocProfile":
        d = np.load(path, allow_pickle=False)
        return cls({str(c): i for i, c in enumerate(d["clue_words"])}, d["b"],
                   json.loads(str(d["word_lists"])))

    def columns(self, clue: str, candidates: list[str]) -> np.ndarray:
        """(len(candidates), len(PROFILE_FEATURES))."""
        n = len(candidates)
        r = self.clue_rows.get(clue.lower())
        b = np.full(len(B_FEATURES), np.nan) if r is None else self.b[r]
        target = clue_forms(assoc_norm(clue.lower()))
        share, rank = np.full(n, np.nan), np.full(n, np.nan)
        for i, w in enumerate(candidates):
            ls = self.word_lists.get(assoc_norm(w))
            if not ls:
                continue
            hits = [next((j + 1 for j, x in enumerate(l) if x in target), 0) for l in ls]
            share[i] = np.mean([h > 0 for h in hits])
            rank[i] = np.mean([1.0 / h if h else 0.0 for h in hits])
        return np.column_stack([np.tile(b, (n, 1)), share, rank, rank_desc(rank)]).astype(np.float64)
