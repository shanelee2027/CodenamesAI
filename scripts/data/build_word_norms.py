"""Per-board-word human norms: concreteness, familiarity, frequency.

**Why word-level priors at all.** The SHAP attribution (notebooks/listener_shap.ipynb)
puts `word_mean_sim` and `word_sd_sim` second and third by contribution per
feature, behind only `z_numberbatch`. Both describe the *word*, with no clue
involved. That says the model wants to know things about a word independent of
what was said about it, and concreteness is the best-attested such property:
a listener told "animal" and looking at LION and SPIRIT has a reason to prefer
the one they can picture.

Source: Brysbaert, Warriner & Kuperman (2014), 39,954 lemmas rated by 4,000+
participants. The same file carries SUBTLEX frequency and percent-known, which
are the other two classic word-level priors, so all three come free.

A fourth prior comes from WordNet rather than the norms: **polysemy**, the
number of senses a board word has. A highly polysemous word is reachable from
many unrelated clues, which is exactly the property that makes a listener
mis-guess, and no similarity or association feature expresses it.

Output `cache/word_norms.npz`: one row per board word, NaN where the word is
not in the norms.

Usage:
    python scripts/data/build_word_norms.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

SRC = PROJECT_ROOT / "data" / "norms" / "Concreteness_ratings_Brysbaert_et_al_BRM.txt"
OUT = PROJECT_ROOT / "cache" / "word_norms.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])

    norms: dict[str, tuple[float, float, float, float]] = {}
    with args.source.open(encoding="utf-8", errors="replace") as fh:
        header = next(fh).rstrip("\n").split("\t")
        ix = {name: i for i, name in enumerate(header)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(header):
                continue
            try:
                norms[p[ix["Word"]].lower()] = (
                    float(p[ix["Conc.M"]]), float(p[ix["Conc.SD"]]),
                    float(p[ix["Percent_known"]]), float(p[ix["SUBTLEX"]]),
                )
            except ValueError:
                continue
    print(f"{len(norms):,} lemmas in the norms")

    from nltk.corpus import wordnet as wn

    out = np.full((len(board_words), 5), np.nan, dtype=np.float32)
    for i, w in enumerate(board_words):
        n_senses = float(len(wn.synsets(w)))
        v = norms.get(w)
        if v is not None:
            # log1p on frequency: SUBTLEX counts span five orders of magnitude,
            # and a tree splitting on the raw count would spend every split in
            # the top decile.
            out[i] = (v[0], v[1], v[2], float(np.log1p(v[3])), n_senses)
        else:
            out[i, 4] = n_senses
    cov = np.isfinite(out[:, 0]).mean()
    print(f"board coverage: {int(cov*len(board_words))}/{len(board_words)} ({cov:.1%})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, norms=out, board_words=np.array(board_words))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
