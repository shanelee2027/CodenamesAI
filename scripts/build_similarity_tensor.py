"""Build the GloVe similarity tensor (SCOPE.md §M2).

Loads GloVe 6B-300d, filters a ~250k-word clue vocabulary, builds vectors
for the board vocabulary (mean-pooling multi-word entries like "New york"
-- GloVe only has single-token vectors, so a phrase's vector is the mean
of its parts' vectors, a standard practice for representing short phrases
with single-word embeddings), normalizes everything to unit vectors, and
computes cosine similarity for every (clue, board word) pair on GPU in
batches. Output is a memory-mapped fp16 tensor of shape
(n_clues, n_board_words, n_spaces=1) plus JSON vocab index files, all in
cache/ (gitignored -- this is a build artifact, regenerate it locally).

GloVe's 6B vocab file is frequency-descending ordered (verified empirically:
first entries are "the", ",", ".", "of" ...; last entries are junk/rare
tokens) -- so "frequency threshold" (SCOPE.md's phrase) is implemented as a
rank cutoff via --vocab-size, rather than pulling in an external word-
frequency source. Filtering to purely lowercase-alphabetic tokens first
(dropping punctuation/digit tokens GloVe also assigns vectors to) and then
taking the top --vocab-size of what's left lands close to SCOPE's ~250k
target using GloVe's own ordering, no new dependency needed.

Usage:
    python scripts/build_similarity_tensor.py
    python scripts/build_similarity_tensor.py --vocab-size 250000 --batch-size 20000
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ALPHABETIC = re.compile(r"^[a-z]+$")


def ensure_glove_extracted(zip_path: Path, extract_dir: Path, filename: str) -> Path:
    dest = extract_dir / filename
    if dest.exists():
        return dest
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(filename, extract_dir)
    return dest


def load_glove(path: Path) -> tuple[list[str], np.ndarray]:
    """Returns (words, vectors) in file order (== frequency-descending rank)."""
    words: list[str] = []
    vectors: list[np.ndarray] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    return words, np.stack(vectors)


def build_clue_vocab(glove_words: list[str], glove_vectors: np.ndarray, vocab_size: int) -> tuple[list[str], np.ndarray]:
    clue_words: list[str] = []
    clue_idxs: list[int] = []
    for i, w in enumerate(glove_words):
        if ALPHABETIC.match(w):
            clue_words.append(w)
            clue_idxs.append(i)
            if len(clue_words) == vocab_size:
                break
    return clue_words, glove_vectors[clue_idxs]


def build_board_vectors(board_words: list[str], glove_index: dict[str, int], glove_vectors: np.ndarray) -> np.ndarray:
    dim = glove_vectors.shape[1]
    out = np.zeros((len(board_words), dim), dtype=np.float32)
    for i, word in enumerate(board_words):
        lw = word.lower()
        if lw in glove_index:
            out[i] = glove_vectors[glove_index[lw]]
            continue
        parts = re.split(r"[\s-]+", lw)
        missing = [p for p in parts if p not in glove_index]
        if missing:
            raise KeyError(
                f"board word {word!r} has no GloVe vector, and part(s) {missing} "
                "also missing -- can't even mean-pool it. Fix the board word list "
                "or extend GloVe coverage before proceeding."
            )
        part_vectors = np.stack([glove_vectors[glove_index[p]] for p in parts])
        out[i] = part_vectors.mean(axis=0)
    return out


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def compute_similarity_tensor(
    clue_vectors: np.ndarray, board_vectors: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    board_t = torch.from_numpy(board_vectors).to(device)  # (n_board, dim)
    n_clues = clue_vectors.shape[0]
    n_board = board_vectors.shape[0]
    result = np.empty((n_clues, n_board), dtype=np.float16)

    for start in range(0, n_clues, batch_size):
        end = min(start + batch_size, n_clues)
        batch = torch.from_numpy(clue_vectors[start:end]).to(device)
        sims = batch @ board_t.T  # (batch, n_board), cosine sim since both sides are unit vectors
        result[start:end] = sims.to(torch.float16).cpu().numpy()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-zip", type=Path, default=Path("data/embeddings/raw/glove.6B.zip"))
    parser.add_argument("--glove-dim", type=int, default=300)
    parser.add_argument("--glove-extract-dir", type=Path, default=Path("data/embeddings/glove"))
    parser.add_argument("--board-words", type=Path, default=Path("codenames/assets/board_words.txt"))
    parser.add_argument("--vocab-size", type=int, default=250_000, help="clue vocabulary size (rank cutoff over GloVe's frequency-ordered vocab)")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("cache"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    glove_filename = f"glove.6B.{args.glove_dim}d.txt"
    glove_path = ensure_glove_extracted(args.glove_zip, args.glove_extract_dir, glove_filename)
    print(f"loading {glove_path} ...")
    glove_words, glove_vectors = load_glove(glove_path)
    glove_index = {w: i for i, w in enumerate(glove_words)}
    print(f"loaded {len(glove_words)} GloVe vectors in {time.time()-t0:.1f}s")

    clue_words, clue_vectors = build_clue_vocab(glove_words, glove_vectors, args.vocab_size)
    print(f"clue vocabulary: {len(clue_words)} words")

    board_words = [l.strip() for l in args.board_words.read_text().splitlines() if l.strip()]
    board_vectors = build_board_vectors(board_words, glove_index, glove_vectors)
    print(f"board vocabulary: {len(board_words)} words")

    clue_vectors = normalize_rows(clue_vectors)
    board_vectors = normalize_rows(board_vectors)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.time()
    sim_2d = compute_similarity_tensor(clue_vectors, board_vectors, device, args.batch_size)
    compute_time = time.time() - t1
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    tensor = sim_2d[:, :, np.newaxis]  # (n_clues, n_board_words, n_spaces=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "similarity_tensor.npy", tensor)
    (args.output_dir / "clue_vocab.json").write_text(json.dumps(clue_words))
    (args.output_dir / "board_vocab.json").write_text(json.dumps(board_words))
    (args.output_dir / "similarity_meta.json").write_text(json.dumps({
        "spaces": ["glove"],
        "shape": list(tensor.shape),
        "glove_source": str(glove_path),
        "vocab_size_arg": args.vocab_size,
    }, indent=2))

    total_time = time.time() - t0
    size_mb = tensor.nbytes / 1e6
    print(f"\ntensor shape: {tensor.shape}, {size_mb:.1f} MB on disk")
    print(f"similarity compute time: {compute_time:.1f}s")
    print(f"peak VRAM: {peak_vram:.2f} GB")
    print(f"total wall time: {total_time:.1f}s")
    print(f"written to {args.output_dir}/")


if __name__ == "__main__":
    main()
