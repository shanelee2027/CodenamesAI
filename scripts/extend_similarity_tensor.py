"""Extend the similarity tensor with Numberbatch and/or Wikipedia2Vec
(SCOPE.md §M4, partial -- fastText is trained on the Fandom corpus, not
covered here since that corpus isn't fully collected yet).

Computes this space's similarity against whatever clue/board vocabulary
scripts/build_similarity_tensor.py already fixed (an intersection across
all three downloaded spaces as of the first-pass revision -- see that
script's docstring). A word not found in this space's source file gets
NaN in this space's slice (see _embedding_lib.compute_similarity) rather
than being dropped or zeroed -- should be rare now that the vocabulary is
an intersection, but not assumed to be zero.

Usage:
    python scripts/extend_similarity_tensor.py --space numberbatch
    python scripts/extend_similarity_tensor.py --space wikipedia2vec
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _embedding_lib import SPACE_CONFIGS, build_vectors_for_vocab, compute_similarity, load_filtered_vectors, normalize_rows, split_words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", choices=[s for s in SPACE_CONFIGS if s != "glove"], required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()

    config = SPACE_CONFIGS[args.space]
    source = args.source or config["default_source"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"space: {args.space}, source: {source}, device: {device}")

    t0 = time.time()
    clue_words: list[str] = json.loads((args.cache_dir / "clue_vocab.json").read_text())
    board_words: list[str] = json.loads((args.cache_dir / "board_vocab.json").read_text())
    meta = json.loads((args.cache_dir / "similarity_meta.json").read_text())
    tensor = np.load(args.cache_dir / "similarity_tensor.npy")  # full load, not mmap -- we're rewriting it

    wanted: set[str] = set(clue_words) | {w.lower() for w in board_words}
    for w in board_words:
        parts = split_words(w)
        if len(parts) > 1:
            wanted.update(parts)
            if config["multiword_join"] is not None:
                wanted.add(config["multiword_join"].join(parts))

    print(f"scanning {source} for {len(wanted)} candidate tokens...")
    vectors = load_filtered_vectors(source, config["opener"], wanted, config["skip_prefixes"], config["has_header"])
    print(f"found {len(vectors)} of {len(wanted)} candidate tokens ({time.time()-t0:.1f}s)")

    dim = next(iter(vectors.values())).shape[0]
    clue_matrix, clue_valid = build_vectors_for_vocab(clue_words, vectors, dim, is_board=False, multiword_join=None)
    print(f"clue coverage: {clue_valid.sum()}/{len(clue_words)} ({100*clue_valid.mean():.1f}%)")

    board_matrix, board_valid = build_vectors_for_vocab(board_words, vectors, dim, is_board=True, multiword_join=config["multiword_join"])
    missing_board = [w for w, ok in zip(board_words, board_valid) if not ok]
    print(f"board coverage: {board_valid.sum()}/{len(board_words)}" + (f", missing: {missing_board}" if missing_board else ""))

    clue_matrix = normalize_rows(clue_matrix, clue_valid)
    board_matrix = normalize_rows(board_matrix, board_valid)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.time()
    sim_2d = compute_similarity(clue_matrix, clue_valid, board_matrix, board_valid, device, args.batch_size)
    compute_time = time.time() - t1
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    new_slice = sim_2d.astype(np.float16)[:, :, np.newaxis]
    spaces: list[str] = meta["spaces"]
    if args.space in spaces:
        idx = spaces.index(args.space)
        tensor[:, :, idx : idx + 1] = new_slice
        print(f"replaced existing '{args.space}' slice at index {idx}")
    else:
        tensor = np.concatenate([tensor, new_slice], axis=-1)
        spaces = spaces + [args.space]
        print(f"appended '{args.space}' as slice index {len(spaces)-1}")

    np.save(args.cache_dir / "similarity_tensor.npy", tensor)
    meta["spaces"] = spaces
    meta["shape"] = list(tensor.shape)
    (args.cache_dir / "similarity_meta.json").write_text(json.dumps(meta, indent=2))

    total_time = time.time() - t0
    print(f"\ntensor shape: {tensor.shape}, spaces: {spaces}")
    print(f"similarity compute time: {compute_time:.1f}s, peak VRAM: {peak_vram:.2f} GB")
    print(f"total wall time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
