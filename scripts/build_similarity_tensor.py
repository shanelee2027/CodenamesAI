"""Build the similarity tensor's clue vocabulary and GloVe slice (SCOPE.md §M2).

**Revision (first-pass simplification, see docs/log.md):** clue vocabulary
is now the INTERSECTION of all three downloaded spaces' own vocabularies,
not their union. An earlier version of this script used only GloVe's
top-N words, which structurally prevented any word GloVe doesn't know from
ever being a candidate clue (the "Technoblade" problem); that was fixed by
switching to a union. The union in turn reintroduced a related problem one
level up: a clue could still be one only *some* guessers know, so a
guesser's failure to guess it might reflect a real vocabulary gap rather
than a genuinely bad clue -- exactly the effect SCOPE's diverse guesser
pool (§3) is designed to average out across many guessers, but which
becomes a real problem for a deliberately small first-pass pool. The
intersection sidesteps it directly: every legal clue has a real vector in
every space that's currently built, so no guesser in the pool is
structurally disadvantaged by coverage gaps. This is a divergence from
SCOPE.md's original union approach -- documented here and in docs/SCOPE.md
and docs/log.md, not silently reverted.

Per-space vocabulary contribution (unchanged from the union version):
  - GloVe and Wikipedia2Vec are both frequency-descending ordered in their
    raw files (verified empirically) -- take each one's top --*-vocab-size
    alphabetic words via a token-only scan that stops early once enough
    are collected (e.g. Wikipedia2Vec: the first 250k qualifying tokens
    appear within the first 12% of its 4.53M lines, so this is much
    faster than a full scan).
  - Numberbatch has no reliable frequency ordering in its file (verified
    empirically -- its first entries are junk tokens like "##") and its
    total alphabetic vocabulary is modest (~359k), so all of it is
    included rather than attempting a rank cutoff.

Every word in the final intersection gets a GloVe similarity value if
GloVe's *full* 400k-word vocabulary has it (not just its top-250k) -- a
word whose presence in the intersection was actually decided by its rank
in Wikipedia2Vec/Numberbatch might still exist further down GloVe's own
list. In practice, since intersection membership already requires a
GloVe-contribution hit, this GloVe slice should end up ~100% covered;
kept as NaN-on-miss for consistency with extend_similarity_tensor.py
rather than assuming that in code.

Usage:
    python scripts/build_similarity_tensor.py
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from _embedding_lib import (
    ALPHABETIC,
    SPACE_CONFIGS,
    all_alphabetic_words,
    build_vectors_for_vocab,
    compute_similarity,
    normalize_rows,
    ranked_alphabetic_words,
)


def ensure_glove_extracted(zip_path: Path, extract_dir: Path, filename: str) -> Path:
    dest = extract_dir / filename
    if dest.exists():
        return dest
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(filename, extract_dir)
    return dest


def load_glove_full(path: Path) -> tuple[list[str], np.ndarray]:
    """Returns (words, vectors) for GloVe's *entire* vocabulary, in file
    order (== frequency rank). Needed as a full dict (not just the top-N
    used to seed the vocab union) since a word contributed by another
    space might exist further down GloVe's list."""
    words: list[str] = []
    vectors: list[np.ndarray] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    return words, np.stack(vectors)


def build_board_words(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-zip", type=Path, default=Path("data/embeddings/raw/glove.6B.zip"))
    parser.add_argument("--glove-dim", type=int, default=300)
    parser.add_argument("--glove-extract-dir", type=Path, default=Path("data/embeddings/glove"))
    parser.add_argument("--numberbatch-source", type=Path, default=SPACE_CONFIGS["numberbatch"]["default_source"])
    parser.add_argument("--wikipedia2vec-source", type=Path, default=SPACE_CONFIGS["wikipedia2vec"]["default_source"])
    parser.add_argument("--board-words", type=Path, default=Path("codenames/assets/board_words.txt"))
    parser.add_argument("--glove-vocab-size", type=int, default=250_000, help="GloVe's own contribution to the vocab union (rank cutoff)")
    parser.add_argument("--wikipedia2vec-vocab-size", type=int, default=250_000, help="Wikipedia2Vec's own contribution to the vocab union (rank cutoff)")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("cache"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    t0 = time.time()

    glove_filename = f"glove.6B.{args.glove_dim}d.txt"
    glove_path = ensure_glove_extracted(args.glove_zip, args.glove_extract_dir, glove_filename)
    print(f"loading full GloVe vocabulary from {glove_path} ...")
    glove_words, glove_vectors = load_glove_full(glove_path)
    glove_index = {w: i for i, w in enumerate(glove_words)}
    print(f"loaded {len(glove_words)} GloVe vectors ({time.time()-t0:.1f}s)")

    print(f"collecting top {args.glove_vocab_size} alphabetic GloVe words...")
    glove_contribution = set()
    for w in glove_words:
        if ALPHABETIC.match(w):
            glove_contribution.add(w)
            if len(glove_contribution) >= args.glove_vocab_size:
                break

    print(f"collecting top {args.wikipedia2vec_vocab_size} alphabetic Wikipedia2Vec words...")
    wiki_config = SPACE_CONFIGS["wikipedia2vec"]
    wiki_contribution = set(ranked_alphabetic_words(
        args.wikipedia2vec_source, wiki_config["opener"], wiki_config["skip_prefixes"],
        wiki_config["has_header"], limit=args.wikipedia2vec_vocab_size,
    ))
    print(f"  found {len(wiki_contribution)}")

    print("collecting all alphabetic Numberbatch words (no reliable rank order to cut by)...")
    nb_config = SPACE_CONFIGS["numberbatch"]
    nb_contribution = all_alphabetic_words(
        args.numberbatch_source, nb_config["opener"], nb_config["skip_prefixes"], nb_config["has_header"],
    )
    print(f"  found {len(nb_contribution)}")

    clue_words = sorted(glove_contribution & wiki_contribution & nb_contribution)
    print(f"clue vocabulary (intersection): {len(clue_words)} words")

    board_words = build_board_words(args.board_words)
    print(f"board vocabulary: {len(board_words)} words")

    dim = glove_vectors.shape[1]
    clue_matrix = np.zeros((len(clue_words), dim), dtype=np.float32)
    clue_valid = np.zeros(len(clue_words), dtype=bool)
    for i, w in enumerate(clue_words):
        idx = glove_index.get(w)
        if idx is not None:
            clue_matrix[i] = glove_vectors[idx]
            clue_valid[i] = True
    print(f"GloVe clue coverage: {clue_valid.sum()}/{len(clue_words)} ({100*clue_valid.mean():.1f}%)")

    board_vectors_dict = {w: glove_vectors[glove_index[w]] for w in glove_index}
    board_matrix, board_valid = build_vectors_for_vocab(
        board_words, board_vectors_dict, dim, is_board=True, multiword_join=None,
    )
    missing_board = [w for w, ok in zip(board_words, board_valid) if not ok]
    print(f"GloVe board coverage: {board_valid.sum()}/{len(board_words)}" + (f", missing: {missing_board}" if missing_board else ""))

    clue_matrix = normalize_rows(clue_matrix, clue_valid)
    board_matrix = normalize_rows(board_matrix, board_valid)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.time()
    sim_2d = compute_similarity(clue_matrix, clue_valid, board_matrix, board_valid, device, args.batch_size)
    compute_time = time.time() - t1
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    tensor = sim_2d.astype(np.float16)[:, :, np.newaxis]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "similarity_tensor.npy", tensor)
    (args.output_dir / "clue_vocab.json").write_text(json.dumps(clue_words))
    (args.output_dir / "board_vocab.json").write_text(json.dumps(board_words))
    (args.output_dir / "similarity_meta.json").write_text(json.dumps({
        "spaces": ["glove"],
        "shape": list(tensor.shape),
        "glove_source": str(glove_path),
        "glove_vocab_size_arg": args.glove_vocab_size,
        "wikipedia2vec_vocab_size_arg": args.wikipedia2vec_vocab_size,
        "vocab_note": "intersection of glove/numberbatch/wikipedia2vec vocabularies (first-pass revision -- see script docstring and docs/log.md)",
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
