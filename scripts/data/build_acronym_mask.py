"""Flag the clue vocabulary's acronyms, so spymasters can refuse to play them.

PHD, GM, CIA and HSBC are all legal one-word clues and the model gave them --
`phd` 29 times, `gm` 20, `rn` 14, `hsbc` 8 across the recorded games. They are
bad clues at a real table and they make the model look worse than it is.

**This is a pool restriction, not a rule of Codenames.** It lives here and is
applied the way `max_rarity` is, rather than in `is_legal_clue`: legality is
"identical for every model" (docs/design-decisions.md) and every recorded
result was produced under the current definition, so moving this into the rules
would silently redefine what those runs measured. A spymaster opting out of a
slice of the pool is a different kind of claim from a clue being forbidden.

Detection, in the order the tests fire:

1. **No vowel at all** (counting y) -- `gm`, `td`, `hm`, `hsbc`. Nothing
   pronounceable looks like this.
2. **Every WordNet spelling is ALL CAPS or dotted** -- `CIA`, `PhD`, `U.S.A.`.
   WordNet keeps canonical casing in its lemma names, which is the signal.
   It has to be *every* spelling, not any: `cat` carries `CAT` (the scan) and
   `led` carries `LED` beside the ordinary words, and requiring only one caps
   spelling flagged `cat`, `pet`, `zip`, `shape` and `led` as acronyms.
   Inflections are followed through `morphy` first, or `limbs`, `gods` and
   `mice` are flagged for having no lemma of their own.
3. **No WordNet entry and at most four letters** -- `nba`, `wwe`, `cpu`. The
   length bound is what keeps `smartphone` and `nintendo`, which are equally
   absent from WordNet, out of it.

Rule 2 is why `laser`, `radar` and `scuba` survive: they are acronyms
historically but lexicalised now, and WordNet spells them in lower case.

Known false positives, all short names WordNet lacks -- `jill`, `joey`,
`cena`. Losing them costs a little; `the` and `when`, flagged by the same
branch, are no loss at all. Measured over every clue ever played: 81 of 3,534
distinct clues, 1.5% of plays.

Usage:
    python scripts/data/build_acronym_mask.py            # -> cache/acronym_mask.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


from codenames.similarity import DEFAULT_CACHE_DIR

VOWEL = re.compile(r"[aeiouy]")
SHORT_UNKNOWN = 4       # no WordNet entry and this short or shorter -> acronym


def _spellings(word: str, wn) -> set[str]:
    """Every WordNet lemma spelling of `word`, following inflections.

    Without `morphy` a plural has no lemma of its own and falls through to the
    unknown-word branch, which flagged `limbs`, `gods` and `mice`.
    """
    candidates = {word}
    for pos in ("n", "v", "a", "r"):
        base = wn.morphy(word, pos)
        if base:
            candidates.add(base)
    out: set[str] = set()
    for cand in candidates:
        for syn in wn.synsets(cand):
            for name in syn.lemma_names():
                if name.lower().replace(".", "").replace("_", "") == cand.lower():
                    out.add(name)
    return out


def is_acronym(word: str, wn) -> bool:
    if len(word) < 2 or not word.isalpha():
        return False
    if not VOWEL.search(word):
        return True
    spellings = _spellings(word, wn)
    if spellings:
        return all(s.isupper() or "." in s for s in spellings)
    return len(word) <= SHORT_UNKNOWN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from nltk.corpus import wordnet as wn

    vocab = json.loads((args.cache_dir / "clue_vocab.json").read_text())
    mask = np.array([is_acronym(w, wn) for w in vocab], dtype=bool)
    out = args.out or (args.cache_dir / "acronym_mask.npz")
    np.savez_compressed(out, mask=mask, clue_words=np.array(vocab, dtype=object))
    flagged = [w for w, m in zip(vocab, mask) if m]
    print(f"{mask.sum()} of {len(vocab)} clue words flagged ({100*mask.mean():.1f}%) -> {out}")
    print("sample:", flagged[:20])


if __name__ == "__main__":
    main()
