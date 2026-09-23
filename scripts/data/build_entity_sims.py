"""Recover the Wikipedia2Vec ENTITY vectors the tensor build throws away.

`scripts/data/_embedding_lib.py` skips every line prefixed `ENTITY/` when
building the similarity tensor, keeping only plain word vectors. That discards
867k+ vectors -- more than the word vectors it keeps -- and with them the
encyclopedic knowledge behind failures like `schmidt -> Scorpion` (the Schmidt
sting pain index) and `japan -> Police` (both new-wave bands). A word vector
for "nikon" encodes how the token is used; the entity vector for the *company*
Nikon sits near Olympus and Canon because the articles do.

**Why a side table rather than a fourth space in the tensor.** The tensor's
vocabulary is deliberately the intersection of its three spaces, so every legal
clue has a real vector everywhere, and `clue_stats` z-scores each clue across
all 400 board words. Entity coverage is partial, so a clue covered for 150 of
400 board words would get its mean and sd from a different subset than a fully
covered clue -- and `expected_words` takes an argmax over 111k clues, so that
scale difference would bias which clues get chosen. The guesser has no such
problem: it compares words within one board for one clue, where missing is
simply missing.

Same-name is not the same sense: Wikipedia resolves `Bat` to a primary topic
(the animal, or Batman) and `Bank` to one of finance or river. An entity vector
carries a disambiguated sense, which is a strength when it is the right one and
a liability otherwise. That is exactly what the learned model is for.

Output `cache/entity_sims.npz`: a dense (n_clue_pool x n_board_words) float32
matrix of cosine similarities, NaN where either side has no entity vector,
plus the clue-vocabulary row index so the feature extractor can look up a row.

Usage:
    python scripts/data/build_entity_sims.py
"""

from __future__ import annotations

import argparse
import bz2
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

SRC = PROJECT_ROOT / "data" / "embeddings" / "raw" / "enwiki_20180420_300d.txt.bz2"
OUT = PROJECT_ROOT / "cache" / "entity_sims.npz"
MAX_RARITY = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=MAX_RARITY,
                    help="only clues the spymaster could actually play")
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]

    # One pass over the file; keep only entities we could ever look up.
    wanted = set(board_words) | set(pool_words)
    print(f"scanning {args.source.name} for ENTITY vectors matching "
          f"{len(wanted):,} words of interest...", flush=True)
    vecs: dict[str, np.ndarray] = {}
    seen = 0
    with bz2.open(args.source, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            if not line.startswith("ENTITY/"):
                continue
            seen += 1
            head, _, rest = line.partition(" ")
            name = head[len("ENTITY/"):].replace("_", " ").lower()
            if name in wanted and name not in vecs:
                v = np.fromstring(rest, dtype=np.float32, sep=" ")
                if v.size:
                    vecs[name] = v
    print(f"  {seen:,} ENTITY lines scanned, {len(vecs):,} matched")

    have_board = [w for w in board_words if w in vecs]
    have_pool = [w for w in pool_words if w in vecs]
    print(f"  board words with an entity vector: {len(have_board)}/{len(board_words)} "
          f"({len(have_board)/len(board_words):.1%})")
    print(f"  clue-pool words with one:          {len(have_pool)}/{len(pool_words)} "
          f"({len(have_pool)/len(pool_words):.1%})")
    if not have_board or not have_pool:
        raise SystemExit("no overlap -- nothing to build")

    B = np.stack([vecs[w] for w in have_board])
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    C = np.stack([vecs[w] for w in have_pool])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)

    out = np.full((len(pool_words), len(board_words)), np.nan, dtype=np.float32)
    row_of = {w: i for i, w in enumerate(pool_words)}
    col_of = {w: i for i, w in enumerate(board_words)}
    rows = np.array([row_of[w] for w in have_pool])
    cols = np.array([col_of[w] for w in have_board])
    out[np.ix_(rows, cols)] = C @ B.T
    print(f"  matrix {out.shape}, {np.isfinite(out).mean():.1%} populated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sims=out,
                        clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board_words))
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
