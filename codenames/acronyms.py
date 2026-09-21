"""Load the precomputed acronym mask over a clue vocabulary.

Built by `scripts/data/build_acronym_mask.py`, which documents the detection
rules and why this is a pool restriction rather than a rule of Codenames.
Kept separate from that script so importing it costs nothing: the artifact is
a list of words, and nothing at game time needs WordNet or nltk.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


@lru_cache(maxsize=4)
def _flagged_words(path: Path) -> frozenset[str]:
    """The flagged words themselves, cached: the arena builds a fresh
    spymaster in every worker process, and both stages of the learned listener
    ask for the same file."""
    data = np.load(path, allow_pickle=True)
    words = np.asarray(data["clue_words"], dtype=object)
    mask = np.asarray(data["mask"], dtype=bool)
    return frozenset(str(w) for w, m in zip(words, mask) if m)


def load_acronym_mask(cache_dir: Path, clue_words: Sequence[str]) -> np.ndarray | None:
    """`(len(clue_words),)` bool, or None if the artifact has not been built.

    Keyed by word rather than by position. An earlier version stored a bare
    positional array and asserted the length matched, which was brittle in
    both directions: it rejected any vocabulary but the exact one it was built
    against (every test injecting a synthetic ClueStats, for one), while still
    only checking length -- two same-sized vocabularies in different orders
    would have banned an arbitrary slice of the pool silently. Looking words up
    cannot be wrong about which word it flagged; a vocabulary the artifact has
    never seen simply comes back all-False.

    None rather than raising when the file is absent, so a fresh checkout still
    plays -- it just plays acronyms until the build script is run.
    """
    path = Path(cache_dir) / "acronym_mask.npz"
    if not path.exists():
        return None
    flagged = _flagged_words(path)
    return np.array([w in flagged for w in clue_words], dtype=bool)
