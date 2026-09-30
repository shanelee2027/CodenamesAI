"""Which sense of the clue each board word connects through, from WordNet.

**Why.** A clue has several readings, and a guesser commits to one: for
"bank" the board may hold both River and Money, and whichever the strong
candidates share is the one the guesser takes. The listener scores each word
against the clue on its own, so a word that is close to the clue in a reading
no other candidate supports is not discounted. The prompt comparison found
the same effect across picks: "a list stays on the reading its first word
committed to" (docs/log.md). So the listener gets a feature for whether a
word shares its best clue sense with the board's other strong candidates
(codenames/listener_features.py, `sense_*`), and a per-pick model can ask
whether it shares the sense of the word already picked.

**What is stored.** Per (clue, board word), the index into
`wn.synsets(clue)` of the clue sense with the highest Wu-Palmer similarity to
any sense of the word (-1 when there is none), and that similarity. The
ancestor-map trick is build_wordnet_sims.py's, kept per clue sense instead of
merged, since merging is exactly what loses the sense.

Output `cache/wordnet_senses.npz`: `sense` (int16) and `sense_wup`
(float32), clue-pool rows x board-word columns.

    python scripts/data/build_wordnet_senses.py --workers 14
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

OUT = PROJECT_ROOT / "cache" / "wordnet_senses.npz"
_D: dict = {}


def sense_map(s) -> dict[int, tuple[int, int]]:
    """{ancestor key -> (distance from this one sense, ancestor depth)}."""
    out: dict[int, tuple[int, int]] = {}
    for anc, dist in s.hypernym_distances():
        k = anc.offset() * 10 + "nvasr".index(anc.pos())
        prev = out.get(k)
        if prev is None or dist < prev[0]:
            out[k] = (dist, anc.max_depth())
    return out


def word_map(word: str) -> dict[int, tuple[int, int]]:
    from nltk.corpus import wordnet as wn

    out: dict[int, tuple[int, int]] = {}
    for s in wn.synsets(word.replace(" ", "_")):
        for k, v in sense_map(s).items():
            if k not in out or v[0] < out[k][0]:
                out[k] = v
    return out


def wup(cm: dict, bm: dict) -> float:
    best = 0.0
    small, large = (cm, bm) if len(cm) < len(bm) else (bm, cm)
    for k, (d1, depth) in small.items():
        o = large.get(k)
        if o is None:
            continue
        den = d1 + o[0] + 2 * depth
        if den:
            best = max(best, 2.0 * depth / den)
    return best


def chunk(args_: tuple[int, int]):
    from nltk.corpus import wordnet as wn

    lo, hi = args_
    boards = _D["board"]
    sense = np.full((hi - lo, len(boards)), -1, dtype=np.int16)
    val = np.full((hi - lo, len(boards)), np.nan, dtype=np.float32)
    for i in range(lo, hi):
        maps = [sense_map(s) for s in wn.synsets(_D["clues"][i])]
        if not maps:
            continue
        for j, bm in enumerate(boards):
            if not bm:
                continue
            scores = [wup(cm, bm) for cm in maps]
            b = int(np.argmax(scores))
            val[i - lo, j] = scores[b]
            if scores[b] > 0:
                sense[i - lo, j] = b
    return lo, sense, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    _D["clues"] = [stats.clue_words[i].lower() for i in pool_rows]
    _D["board"] = [word_map(w) for w in board]
    n = len(_D["clues"])
    step = max(1, n // (args.workers * 8))
    sense = np.full((n, len(board)), -1, dtype=np.int16)
    val = np.full((n, len(board)), np.nan, dtype=np.float32)
    with ProcessPoolExecutor(args.workers) as ex:
        for lo, s, v in ex.map(chunk, [(i, min(i + step, n)) for i in range(0, n, step)]):
            sense[lo:lo + len(s)] = s
            val[lo:lo + len(v)] = v
    have = sense >= 0
    print(f"pairs with a sense: {have.mean():.1%}; clues with >1 sense used on some board word: "
          f"{np.mean([len(set(r[r >= 0].tolist())) > 1 for r in sense]):.1%}")
    np.savez_compressed(args.out, sense=sense, sense_wup=val, clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board))
    print(f"saved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
