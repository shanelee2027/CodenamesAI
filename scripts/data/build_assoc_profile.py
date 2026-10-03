"""Build cache/assoc_profile.npz, the table behind the clue-profile and
reverse-association listener features (codenames/assoc_profile.py).

- **B per clue,** for every clue in ClueStats (every clue extract() can
  score): its association lists' vagueness, rarity percentile, Brysbaert
  concreteness / percent known / log frequency, and WordNet sense count.
- **The association lists of every board word,** for C. All 400 board words
  have lists (collect_associations.py --board), so no gpt-oss call is needed.

Free: reads cache/associations.db, cache/clue_stats.npz, the norms file and
WordNet.

    python scripts/data/build_assoc_profile.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from codenames.assoc_profile import B_FEATURES, vagueness  # noqa: E402

NORMS = PROJECT_ROOT / "data" / "norms" / "Concreteness_ratings_Brysbaert_et_al_BRM.txt"
OUT = PROJECT_ROOT / "cache" / "assoc_profile.npz"


def main() -> None:
    from nltk.corpus import wordnet as wn

    from codenames.clue_policy import PolicyFeatures
    from codenames.clue_stats import ClueStats
    from codenames.listener_training import ASSOCIATIONS, _assoc_norm
    from codenames.similarity import DEFAULT_CACHE_DIR

    lists: dict[str, list[list[str]]] = {}
    conn = sqlite3.connect(f"file:{ASSOCIATIONS}?mode=ro", uri=True)
    for cue, words in conn.execute("SELECT clue, words FROM lists"):
        ws = [_assoc_norm(w) for w in json.loads(words or "[]")]
        if ws:
            lists.setdefault(_assoc_norm(cue), []).append(ws)
    conn.close()

    norms: dict[str, tuple[float, float, float]] = {}
    for line in NORMS.read_text().splitlines()[1:]:
        f = line.split("\t")
        try:
            norms[f[0].lower()] = (float(f[2]), float(f[6]), np.log10(float(f[7]) + 1.0))
        except (ValueError, IndexError):
            continue

    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    clues = [w.lower() for w in stats.clue_words]
    rarity = {w.lower(): float(r) for w, r in zip(stats.clue_words, stats.rarity_percentile)}
    b = np.full((len(clues), len(B_FEATURES)), np.nan)
    for i, c in enumerate(clues):
        conc, known, logfreq = norms.get(c, (np.nan, np.nan, np.nan))
        b[i] = (*vagueness(lists.get(_assoc_norm(c), [])), rarity.get(c, np.nan), conc, known, logfreq,
                float(len(wn.synsets(c.replace(" ", "_")))))

    board = PolicyFeatures.load().board_index          # the 400 board words
    word_lists = {_assoc_norm(w): lists[_assoc_norm(w)] for w in board if _assoc_norm(w) in lists}
    missing = sorted(w for w in board if _assoc_norm(w) not in lists)
    np.savez_compressed(OUT, clue_words=np.array(clues), b=b, word_lists=np.array(json.dumps(word_lists)))
    print(f"{len(clues)} clues ({np.isfinite(b[:, 0]).sum()} with association lists); "
          f"{len(word_lists)} board words with lists, {len(missing)} without{': ' + ', '.join(missing) if missing else ''}"
          f" -> {OUT}")


if __name__ == "__main__":
    main()
