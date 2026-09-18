"""Score every (clue, board word) pair with a cross-encoder, on the GPU.

**Why this is not a fourth embedding space.** glove, numberbatch and wiki2vec
all answer the same question -- cosine between two vectors built independently
of each other -- and the measured pattern in docs/log.md is that another view
of distributional similarity buys nothing (Tier-1 redundant, cohesion null,
entity ~2% of SHAP, raw rivals negative) while a genuinely different kind of
evidence (SWOW human association) buys something. A cross-encoder is different
in kind: it puts both words through one transformer *together*, so the
representation of each is conditioned on the other. That is structurally closer
to what the LLM listener itself does than any static vector can be.

Output `cache/crossenc_sims.npz`: a dense (n_clue_pool x n_board_words) float32
matrix, plus the clue-vocabulary row index, matching the layout of
cache/entity_sims.npz so the feature extractor treats it the same way.

Usage:
    python scripts/data/build_crossencoder_sims.py --batch 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

OUT = PROJECT_ROOT / "cache" / "crossenc_sims.npz"
MODEL = "cross-encoder/stsb-roberta-large"
MAX_RARITY = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-rarity", type=float, default=MAX_RARITY)
    ap.add_argument("--limit-clues", type=int, default=0, help="debug: only score this many clues")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    if args.limit_clues:
        pool_rows = pool_rows[: args.limit_clues]
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(pool_words):,} clues x {len(board_words)} board words "
          f"= {len(pool_words)*len(board_words):,} pairs on {dev}", flush=True)
    model = CrossEncoder(args.model, device=dev, max_length=16)
    # fp16 roughly halves the time and the scores are a feature, not a
    # likelihood -- precision below ~1e-3 is irrelevant here.
    if dev == "cuda":
        model.model.half()

    out = np.empty((len(pool_words), len(board_words)), dtype=np.float32)
    t0 = time.time()
    for i, clue in enumerate(pool_words):
        pairs = [(clue, w) for w in board_words]
        out[i] = model.predict(pairs, batch_size=args.batch,
                               show_progress_bar=False, convert_to_numpy=True)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  {i+1:,}/{len(pool_words):,} clues  {rate:.1f} clues/s  "
                  f"eta {(len(pool_words)-i-1)/rate/60:.1f} min", flush=True)

    print(f"scored in {(time.time()-t0)/60:.1f} min  "
          f"range [{out.min():.3f}, {out.max():.3f}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sims=out, clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board_words))
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
