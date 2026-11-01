"""Similarity side table from a modern text-embedding model.

**Why.** Every embedding space in the tensor and in extra_sims is a static
word-vector model from 2013-2019 (GloVe, Numberbatch, Wikipedia2Vec,
fastText). They give a multi-word name like "Robin Hood" either a phrase
token or the mean of its words' vectors, which for a name is close to
meaningless (bird plus garment). A current text-embedding model encodes any
string, so a name, a compound and a plain word all get a vector of their
own. The model is run once, here, over the clue pool and the board words;
play only reads the table, so no model is loaded and no GPU is used live.

**Normalised over the online game's 800 words.** The stored value is the
cosine z-scored per clue over the online game's deal pool
(data/word_packs/codenames_online.txt), the board-word population a model
trained on the online split sees. No existing model reads this table, so
there is no older footing to keep.

Output (`--out`, default cache/text_sims.npz): `text` (n_clue_pool x
n_board_words) float32, `clue_rows`, `board_words`, and the model name, in
the layout ExtraSims reads.

    HF_HOME=data/hf_home python scripts/data/build_text_sims.py --model BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from codenames.board import ONLINE_WORDLIST_PATH, load_wordlist
from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "cache" / "text_sims.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a sentence-transformers model on the Hugging Face hub")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0, help="the clue pool, as everywhere else")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]
    online = {w.lower() for w in load_wordlist(ONLINE_WORDLIST_PATH)}
    ref = np.array([w.lower() in online for w in board_words])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    enc = lambda ws: model.encode([w.lower() for w in ws], batch_size=args.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    C, B = enc(pool_words), enc(board_words)
    cos = C @ B.T
    mu = cos[:, ref].mean(axis=1, keepdims=True)
    sd = cos[:, ref].std(axis=1, keepdims=True)
    z = ((cos - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    np.savez_compressed(args.out, text=z, clue_rows=pool_rows.astype(np.int32), board_words=np.array(board_words),
                        model=np.array(args.model))
    print(f"{args.model}: dim {C.shape[1]}, {len(pool_words):,} clues x {len(board_words)} board words, "
          f"z over {int(ref.sum())} online words -> {args.out}")


if __name__ == "__main__":
    main()
