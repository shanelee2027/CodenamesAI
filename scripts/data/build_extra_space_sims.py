"""Similarity side-tables for embedding spaces outside the main tensor.

**Why side tables rather than a fourth and fifth slot in the tensor.** The
tensor's vocabulary is deliberately the intersection of its three spaces, so
every legal clue has a real vector everywhere and `clue_stats` can z-score a
clue across all 400 board words on a common footing. Adding a space to the
tensor would shrink that intersection and change which clues are legal, which
silently invalidates every cached rollout and every number in docs/log.md. A
side table changes nothing: the listener compares words within one board for
one clue, where a missing space is simply missing.

**Already z-scored on disk.** The stored value is
`(cos(clue, word) - mean_w cos(clue, w)) / sd_w cos(clue, w)` over all 400
board words, matching exactly what `extract` computes for the tensor spaces via
`clue_stats`. Doing it at build time means the feature extractor cannot
accidentally normalise over the candidate subset instead of the full board,
which would make the feature depend on how much of the board is revealed.

Spaces:
  glove840  -- GloVe Common Crawl 840B, 2.2M cased tokens. The tensor uses
               glove.6B (Wikipedia+Gigaword, 400k), the weakest GloVe release;
               this is the strongest one. GloVe itself has had no new release
               since 2014, so this is a bigger model, not a newer one.
  fasttext  -- fastText Common Crawl, 2M words, subword-based. The practical
               successor to GloVe, and the only space here that can build a
               vector for a word it never saw.

Output `cache/extra_sims.npz`, one (n_clue_pool x n_board_words) float32 matrix
per space, NaN where either side is out of vocabulary.

Usage:
    python scripts/data/build_extra_space_sims.py
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

RAW = PROJECT_ROOT / "data" / "embeddings" / "raw"
OUT = PROJECT_ROOT / "cache" / "extra_sims.npz"
SOURCES = {
    "glove840": (RAW / "glove.840B.300d.zip", "glove.840B.300d.txt"),
    "fasttext": (RAW / "crawl-300d-2M.vec.zip", "crawl-300d-2M.vec"),
}


def read_vectors(zip_path: Path, member: str, wanted: set[str]) -> dict[str, np.ndarray]:
    """One streaming pass, keeping only words we could ever look up.

    Lower-cased on the way in, first spelling wins. GloVe 840B is cased and
    lists more frequent spellings first, so "Apple" before "apple" means the
    proper noun's vector is the one kept -- which is the right call for a game
    whose board words are printed in caps and read as either.
    """
    vecs: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for raw in fh:
                line = raw.decode("utf-8", "replace")
                head, _, rest = line.partition(" ")
                w = head.lower()
                if w not in wanted or w in vecs:
                    continue
                v = np.fromstring(rest, dtype=np.float32, sep=" ")
                if v.size >= 100:
                    vecs[w] = v
    return vecs


def build(vecs: dict[str, np.ndarray], clue_words: list[str], board_words: list[str]) -> np.ndarray:
    have_c = [w for w in clue_words if w in vecs]
    have_b = [w for w in board_words if w in vecs]
    print(f"    clues {len(have_c):,}/{len(clue_words):,} ({len(have_c)/len(clue_words):.1%})"
          f"   board {len(have_b)}/{len(board_words)} ({len(have_b)/len(board_words):.1%})")
    if not have_c or not have_b:
        raise SystemExit("no overlap")

    C = np.stack([vecs[w] for w in have_c])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    B = np.stack([vecs[w] for w in have_b])
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)

    out = np.full((len(clue_words), len(board_words)), np.nan, dtype=np.float32)
    ci = np.array([clue_words.index(w) for w in have_c])
    bi = np.array([board_words.index(w) for w in have_b])
    out[np.ix_(ci, bi)] = C @ B.T

    # z-score each clue row over the board words it does cover.
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(out, axis=1, keepdims=True)
        sd = np.nanstd(out, axis=1, keepdims=True)
    bad = ~np.isfinite(sd) | (sd <= 0)
    sd = np.where(bad, 1.0, sd)
    z = (out - mu) / sd
    z[np.repeat(bad, out.shape[1], axis=1)] = np.nan
    return z.astype(np.float32)


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
    wanted = set(pool_words) | set(board_words)

    payload: dict[str, np.ndarray] = {
        "clue_rows": pool_rows.astype(np.int32),
        "board_words": np.array(board_words),
    }
    for name, (path, member) in SOURCES.items():
        if not path.exists():
            print(f"{name}: {path.name} not found, skipping")
            continue
        print(f"{name}: scanning {path.name} for {len(wanted):,} words...", flush=True)
        vecs = read_vectors(path, member, wanted)
        print(f"  matched {len(vecs):,}")
        payload[name] = build(vecs, pool_words, board_words)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
