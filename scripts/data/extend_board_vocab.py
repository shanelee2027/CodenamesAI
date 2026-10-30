"""Add board words to the similarity tensor without changing the existing ones.

The tensor's board axis is the 400 base-game words. Online boards also draw
from Codenames Duet's 400 (399 new to us), so a board from the official site
can hold words the model has no features for. This appends columns for the
new words, in all three spaces, after the existing 400.

**The existing columns are recomputed, not trusted, and must come out equal.**
Each space's similarities are computed for the old and new board words
together, with the same lookups as the original builds (`_embedding_lib`),
and the old 400 columns are compared against the tensor on disk before
anything is written. A mismatch means this script does not reproduce the
original build, and it stops: appended columns from a different recipe would
be features the boosters were never fitted on.

**Nothing that normalises over the board axis is rebuilt from the new
columns.** `clue_stats` (each clue's mean and sd over the board words) stays
as it is, over the base 400, so every existing z-score is unchanged and a new
word is z-scored on the same footing. The side tables are rebuilt by their
own scripts afterwards (docs/log.md, 2026-10-30).

    python scripts/data/extend_board_vocab.py --words data/word_packs/duet.txt
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
    parser.add_argument("--words", type=Path, required=True, help="new board words, one per line")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clue_words: list[str] = json.loads((args.cache_dir / "clue_vocab.json").read_text())
    board_words: list[str] = json.loads((args.cache_dir / "board_vocab.json").read_text())
    meta = json.loads((args.cache_dir / "similarity_meta.json").read_text())
    tensor = np.load(args.cache_dir / "similarity_tensor.npy")

    have = {w.lower() for w in board_words}
    new = [w for w in (l.strip() for l in args.words.read_text().splitlines()) if w and w.lower() not in have]
    if not new:
        print("no new words")
        return
    words = board_words + new
    print(f"{len(board_words)} board words + {len(new)} new, {len(clue_words):,} clues, spaces {meta['spaces']}")

    slices = []
    for si, space in enumerate(meta["spaces"]):
        t0 = time.time()
        config = SPACE_CONFIGS[space]
        wanted: set[str] = set(clue_words) | {w.lower() for w in words}
        for w in words:
            parts = split_words(w)
            if len(parts) > 1:
                wanted.update(parts)
                if config["multiword_join"] is not None:
                    wanted.add(config["multiword_join"].join(parts))
        vectors = load_filtered_vectors(config["default_source"], config["opener"], wanted,
                                        config["skip_prefixes"], config["has_header"])
        dim = next(iter(vectors.values())).shape[0]
        clue_matrix, clue_valid = build_vectors_for_vocab(clue_words, vectors, dim, is_board=False, multiword_join=None)
        board_matrix, board_valid = build_vectors_for_vocab(words, vectors, dim, is_board=True,
                                                            multiword_join=config["multiword_join"])
        sim = compute_similarity(normalize_rows(clue_matrix, clue_valid), clue_valid,
                                 normalize_rows(board_matrix, board_valid), board_valid,
                                 device, args.batch_size).astype(np.float16)

        old = tensor[:, :, si]
        ours = sim[:, : len(board_words)]
        same = (old == ours) | (np.isnan(old) & np.isnan(ours))
        missing = [w for w, ok in zip(new, board_valid[len(board_words):]) if not ok]
        print(f"{space}: old columns reproduced {same.mean():.6%}, max |diff| "
              f"{np.nanmax(np.abs(old.astype(np.float32) - ours.astype(np.float32))):.2e}; new words without "
              f"a vector: {missing or 'none'} ({time.time() - t0:.0f}s)", flush=True)
        if not same.all():
            raise SystemExit(f"{space}: this build does not reproduce the tensor on disk; nothing written")
        slices.append(sim[:, len(board_words):])

    out = np.concatenate([tensor, np.stack(slices, axis=-1)], axis=1)
    # A running play server memory-maps the tensor: writing over the file in
    # place would change the pages under it. A new file renamed into place
    # leaves the old one mapped until that server restarts.
    tmp = args.cache_dir / "similarity_tensor.tmp.npy"
    np.save(tmp, out)
    tmp.replace(args.cache_dir / "similarity_tensor.npy")
    (args.cache_dir / "board_vocab.json").write_text(json.dumps(words))
    meta["shape"] = list(out.shape)
    meta["board_note"] = f"first {len(board_words)} board words are the base game's; the rest from {args.words}"
    (args.cache_dir / "similarity_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"tensor {tensor.shape} -> {out.shape}")


if __name__ == "__main__":
    main()
