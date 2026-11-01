"""Build `cache/clue_stats.npz` + `cache/clue_stats_meta.json`: each
clue's mean and standard deviation of cosine similarity across all 400
board words, per embedding space, plus its rarity percentile
(`codenames.clue_search.clue_rarity_percentile`).

Why this is a separate build-time artifact rather than something computed
inside a spymaster: `codenames/spymasters/expected_words.py` needs to
z-score a clue's similarity to each board word -- how unusual this
particular board word is for this clue, relative to how the clue behaves
in general -- which needs the clue's own mean/std over the *entire* board
vocabulary. Computing that from the similarity tensor means one pass over
every (clue, board word, space) triple; doing that inside every spymaster
instance (and `codenames/arena.py` spawns one per worker process) would
multiply an identical, model-independent computation by the worker
count. See `codenames/clue_stats.py`'s module docstring for why the
z-scored tensor itself is *not* also cached.

Processes one space at a time (rather than materializing the whole
`(111440, 400, 3)` tensor as float32 at once) to keep peak memory near
one space's slice (~170MB) instead of ~530MB.

Usage:
    python scripts/data/build_clue_stats.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from codenames.clue_search import clue_rarity_percentile
from codenames.clue_stats import clue_vocab_fingerprint
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor


def build_clue_stats(sims: SimilarityTensor, words: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(mean, std, rarity_percentile)`, indexed by `sims.clue_words`
    order. `mean`/`std` are `(n_clues, n_spaces)` float32, each clue's
    similarity statistics across the board words `words` (default every
    board word in the tensor); `rarity_percentile` is `(n_clues,)` float32."""
    cols = None if words is None else np.array(sorted(sims.board_index[w.lower()] for w in words))
    n_clues = len(sims.clue_words)
    n_spaces = len(sims.spaces)
    mean = np.empty((n_clues, n_spaces), dtype=np.float32)
    std = np.empty((n_clues, n_spaces), dtype=np.float32)
    for si in range(n_spaces):
        # One space's full (n_clues, n_board_words) slice, materialized as
        # float32 -- bounded, unlike loading every space at once.
        column = np.asarray(sims.tensor[:, :, si], dtype=np.float32)
        if cols is not None:
            # A contiguous run (the base 400 are columns 0-399) is sliced, not
            # copied: float32 sums run in a different order over a copy, and the
            # stored base statistics are reproduced bit for bit only this way.
            contiguous = np.array_equal(cols, np.arange(cols[0], cols[0] + len(cols)))
            column = column[:, cols[0]:cols[0] + len(cols)] if contiguous else column[:, cols]
        with np.errstate(invalid="ignore"):
            mean[:, si] = np.nanmean(column, axis=1)
            std[:, si] = np.nanstd(column, axis=1)

    rarity = clue_rarity_percentile(sims.clue_words)
    rarity_percentile = np.asarray([rarity[w] for w in sims.clue_words], dtype=np.float32)
    return mean, std, rarity_percentile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--words", choices=("base", "online"), default="base",
                        help="the board words the statistics are taken over: the base game's 400 "
                             "(cache/clue_stats.npz, what every model before the online split reads) "
                             "or the online game's 800 (cache/clue_stats_online.npz). Not the whole "
                             "tensor: it holds every word appended since, and statistics over it "
                             "would move every existing z-score")
    args = parser.parse_args()
    from codenames.board import ONLINE_WORDLIST_PATH, load_wordlist

    words = load_wordlist() if args.words == "base" else load_wordlist(ONLINE_WORDLIST_PATH)
    stem = "clue_stats" if args.words == "base" else "clue_stats_online"

    t0 = time.time()
    sims = SimilarityTensor.load(cache_dir=args.cache_dir)
    print(f"loaded similarity tensor: {len(sims.clue_words)} clues x {len(sims.board_words)} board words x {len(sims.spaces)} spaces")

    mean, std, rarity_percentile = build_clue_stats(sims, words)
    print(f"computed mean/std/rarity in {time.time() - t0:.1f}s")

    np.savez(
        args.cache_dir / f"{stem}.npz",
        mean=mean,
        std=std,
        rarity_percentile=rarity_percentile,
    )
    meta = {
        "clue_vocab_hash": clue_vocab_fingerprint(sims.clue_words),
        "spaces": sims.spaces,
        "n_clues": len(sims.clue_words),
        "n_board_words": len(words),
        "board_words": args.words,
    }
    (args.cache_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.cache_dir / stem}.npz and {stem}_meta.json in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
