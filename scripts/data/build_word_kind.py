"""Build cache/word_kind.npz: per board word, the three columns described in
codenames/word_kind.py (proper_share, hub_assoc, hub_z). No API calls.

Reads the similarity tensor and clue_stats (hub_z), cache/assoc_sims.npz
(hub_assoc) and WordNet from data/nltk_data (proper_share; multi-word names
joined with underscores, as WordNet spells them).

    NLTK_DATA=data/nltk_data python scripts/data/build_word_kind.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.word_kind import KIND_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "cache" / "word_kind.npz"
Z_HUB = 2.0


def proper_share(word: str) -> float:
    from nltk.corpus import wordnet as wn

    syn = wn.synsets(word.replace(" ", "_").replace("-", "_"))
    return float(np.mean([bool(s.instance_hypernyms()) for s in syn])) if syn else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0, help="the clue pool, as everywhere else")
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)

    # hub_z: per space, the share of pool clues with z >= 2 for the word.
    t = np.asarray(sims.tensor[pool], dtype=np.float32)              # (n_pool, n_board, n_spaces)
    mu, sd = stats.mean[pool][:, None, :], stats.std[pool][:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (t - mu) / sd
    hub_z = np.nanmean(np.where(np.isfinite(z), z >= Z_HUB, np.nan), axis=0).mean(axis=1)

    # hub_assoc: the share of pool clues whose lists name the word at all.
    a = np.load(PROJECT_ROOT / "cache" / "assoc_sims.npz")
    assert [str(w).lower() for w in a["board_words"]] == [w.lower() for w in board_words], "board axes differ"
    share = a["assoc_share"]
    hub_assoc = np.nanmean(np.where(np.isfinite(share), share > 0, np.nan), axis=0)

    proper = np.array([proper_share(w) for w in board_words])
    values = np.column_stack([proper, hub_assoc, hub_z]).astype(np.float32)
    assert values.shape[1] == len(KIND_FEATURES)
    np.savez_compressed(args.out, board_words=np.array(board_words), values=values)
    for i, name in enumerate(KIND_FEATURES):
        v = values[:, i]
        print(f"  {name:13s} coverage {np.isfinite(v).mean():6.1%}  mean {np.nanmean(v):.4f}  "
              f"max {np.nanmax(v):.4f} ({board_words[int(np.nanargmax(v))]})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
