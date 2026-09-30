"""Directional is-a membership between every clue and every board word, from
WordNet's hypernym tree.

**Why the existing WordNet feature is not enough.** Wu-Palmer similarity
(scripts/data/build_wordnet_sims.py) is symmetric: it scores how deep the
closest shared ancestor is. Moon and "planet" share "celestial body", and Pie
and "fruit" share "food", so both score well, and the category probe
(scripts/tools/category_probe.py) found the listener ranking Pie above real
fruit for "fruit". The question a listener told "fruit 3" answers is
directional: is this word a KIND OF fruit? That is a path upward from the
board word to the clue, which Pie does not have and Orange does.

Two tables, both on the noun hierarchy, following hypernyms and instance
hypernyms (London is an INSTANCE of city, not a subclass, and would otherwise
be missed):
- `isa`: the board word is a kind of the clue (Octopus -> animal).
  1 / (1 + d), with d the fewest steps from any sense of the word up to any
  sense of the clue; 1 for a shared synset (automobile/Car).
- `isa_rev`: the clue is a kind of the board word (poodle -> Dog), the same
  way round the other direction.

0 means both are in WordNet as nouns and there is no path; NaN means one of
them is not, so there is no evidence either way. Senses are pooled by taking
the shortest path, as build_wordnet_sims does, and for the same reason: a
listener acts on a link whichever sense it runs through. Distance keeps the
stretches honest: Berlin reaches "country" only through its "area" sense, at
d = 5, against d = 2 for China.

Output `cache/isa_sims.npz`, laid out like cache/wordnet_sims.npz (rows are
the clue pool at rarity <= --max-rarity).

    python scripts/data/build_isa_sims.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

OUT = PROJECT_ROOT / "cache" / "isa_sims.npz"


def noun_synsets(word: str) -> list:
    from nltk.corpus import wordnet as wn

    return wn.synsets(word.replace(" ", "_").replace("-", "_"), pos="n")


def ancestors(word: str) -> dict:
    """{synset -> fewest steps up from any noun sense of `word`}, the senses
    themselves at 0."""
    out: dict = {}
    frontier = noun_synsets(word)
    for s in frontier:
        out[s] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for s in frontier:
            for h in s.hypernyms() + s.instance_hypernyms():
                if h not in out:
                    out[h] = d
                    nxt.append(h)
        frontier = nxt
    return out


def path_score(below: dict, above: list) -> float:
    """1 / (1 + fewest steps) from the `below` word's ancestor map up to any
    of `above`'s senses; 0 if none is an ancestor; NaN if either side has no
    noun sense."""
    if not below or not above:
        return np.nan
    d = min((below[s] for s in above if s in below), default=None)
    return 0.0 if d is None else 1.0 / (1.0 + d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]

    b_anc = [ancestors(w) for w in board_words]
    b_syn = [noun_synsets(w) for w in board_words]
    isa = np.full((len(pool_words), len(board_words)), np.nan, dtype=np.float32)
    rev = np.full_like(isa, np.nan)
    for i, c in enumerate(pool_words):
        c_syn = noun_synsets(c)
        if not c_syn:
            continue
        c_anc = ancestors(c)
        for j in range(len(board_words)):
            isa[i, j] = path_score(b_anc[j], c_syn)
            rev[i, j] = path_score(c_anc, b_syn[j])
        if i % 2000 == 0:
            print(f"  {i:,}/{len(pool_words):,}", flush=True)

    print(f"nouns in WordNet: board {np.mean([bool(s) for s in b_syn]):.1%}   "
          f"clues {np.mean(np.isfinite(isa).any(1)):.1%}")
    print(f"pairs with a path: isa {np.nanmean(isa > 0):.2%}   isa_rev {np.nanmean(rev > 0):.2%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, isa=isa, isa_rev=rev, clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board_words))
    print(f"saved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
