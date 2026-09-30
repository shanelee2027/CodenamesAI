"""gpt-oss's free associations to each clue, as a clue x board-word table.

Source: cache/associations.db (scripts/data/collect_associations.py --pool),
5 sampled lists of 25 words per clue, with the clue shown alone. Words are
matched to board words with listener_training's rule (case, accents and
punctuation ignored; a plural or singular form counts).

- `assoc_share`: the share of the clue's lists naming the board word.
- `assoc_rank`: the mean over lists of 1 / position (1-based), 0 where the
  list does not name it. Named first counts far more than named 25th.

NaN for a clue with no lists, so "never asked" is not read as "named by no
list". Rows: the clue pool at rarity <= --max-rarity, like the other tables.

    python scripts/data/build_assoc_sims.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.listener_training import ASSOCIATIONS, _assoc_norm  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

OUT = PROJECT_ROOT / "cache" / "assoc_sims.npz"


def forms(word: str) -> set[str]:
    n = _assoc_norm(word)
    return {n, n + "s", n + "es"} | ({n[:-1]} if n.endswith("s") else set())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ASSOCIATIONS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    args = ap.parse_args()

    lists: dict[str, list[list[str]]] = {}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    for clue, words in conn.execute("SELECT clue, words FROM lists"):
        lst = [_assoc_norm(w) for w in json.loads(words)]
        if lst:
            lists.setdefault(clue.lower(), []).append(lst)
    conn.close()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    bforms = [forms(w) for w in board]
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    clues = [stats.clue_words[i].lower() for i in pool_rows]

    share = np.full((len(clues), len(board)), np.nan, dtype=np.float32)
    rank = np.full_like(share, np.nan)
    for i, c in enumerate(clues):
        ls = lists.get(c)
        if not ls:
            continue
        pos = [{w: k + 1 for k, w in reversed(list(enumerate(lst)))} for lst in ls]
        for j, fs in enumerate(bforms):
            hits = [min((p[f] for f in fs if f in p), default=None) for p in pos]
            share[i, j] = sum(h is not None for h in hits) / len(ls)
            rank[i, j] = float(np.mean([1.0 / h if h else 0.0 for h in hits]))
    covered = np.isfinite(share).any(1)
    print(f"clues with lists {covered.mean():.1%}; board words named for a covered clue: "
          f"{np.nanmean(share > 0):.2%} of pairs")
    np.savez_compressed(args.out, assoc_share=share, assoc_rank=rank,
                        clue_rows=pool_rows.astype(np.int32), board_words=np.array(board))
    print(f"saved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
