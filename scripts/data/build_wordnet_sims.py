"""Wu-Palmer taxonomic similarity between every clue and every board word.

**Why this is a different kind of evidence.** Embeddings measure distributional
similarity and SWOW/PMI measure association. WordNet measures neither: it is a
hand-built is-a hierarchy, so it knows that a LION and a WHALE are both mammals
even if no corpus ever puts them in the same sentence and no one free-associates
one from the other. That taxonomic axis is exactly what a clue like "animal" or
"metal" is trading on.

**Sense ambiguity is handled by taking the max, not by disambiguating.**
WordNet has 18 senses of "bank" and nothing here can tell which one the
spymaster meant. Taking the best-scoring sense pair is the standard treatment
and it is the right one for this task: a listener who sees a link between the
clue and the word acts on it whether or not it was the intended sense -- that
is precisely how Codenames goes wrong.

**Implementation.** Rather than calling `wup_similarity` per pair (40M+ calls),
each word gets one merged ancestor map {synset -> min distance over its senses},
built once. A pair's score is then the best common ancestor under the Wu-Palmer
formula, which is a set intersection. Also emitted is the depth of that best
ancestor: "both are dogs" (depth 13) and "both are entities" (depth 1) are very
different pieces of evidence and the ratio alone does not separate them.

Output `cache/wordnet_sims.npz`, laid out like cache/extra_sims.npz.

Usage:
    python scripts/data/build_wordnet_sims.py --workers 14
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

OUT = PROJECT_ROOT / "cache" / "wordnet_sims.npz"
_D: dict = {}


def ancestor_map(word: str) -> dict[int, tuple[int, int]]:
    """{synset id -> (min distance from `word`, that synset's depth)}.

    Merged across every sense of the word, keeping the shortest route to each
    ancestor -- which is what taking the max over sense pairs amounts to once
    the per-pair maximisation is pushed inside.
    """
    from nltk.corpus import wordnet as wn

    out: dict[int, tuple[int, int]] = {}
    for s in wn.synsets(word):
        for anc, dist in s.hypernym_distances():
            k = anc.offset() * 10 + "nvasr".index(anc.pos()) if anc.pos() in "nvasr" else anc.offset() * 10
            prev = out.get(k)
            if prev is None or dist < prev[0]:
                out[k] = (dist, anc.max_depth())
    return out


def score_chunk(args_: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    lo, hi = args_
    clue_maps, board_maps = _D["clue"], _D["board"]
    wup = np.full((hi - lo, len(board_maps)), np.nan, dtype=np.float32)
    lcs = np.full((hi - lo, len(board_maps)), np.nan, dtype=np.float32)
    for i in range(lo, hi):
        cm = clue_maps[i]
        if not cm:
            continue
        row_w, row_d = wup[i - lo], lcs[i - lo]
        for j, bm in enumerate(board_maps):
            if not bm:
                continue
            best, best_depth = 0.0, 0.0
            small, large = (cm, bm) if len(cm) < len(bm) else (bm, cm)
            for k, (d1, depth) in small.items():
                other = large.get(k)
                if other is None:
                    continue
                d2 = other[0]
                denom = d1 + d2 + 2 * depth
                if denom:
                    v = 2.0 * depth / denom
                    if v > best:
                        best, best_depth = v, float(depth)
            row_w[j], row_d[j] = best, best_depth
    return lo, wup, lcs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]

    print(f"building ancestor maps for {len(pool_words):,} clues + {len(board_words)} board words...",
          flush=True)
    _D["board"] = [ancestor_map(w) for w in board_words]
    _D["clue"] = [ancestor_map(w) for w in pool_words]
    bc = sum(1 for m in _D["board"] if m) / len(board_words)
    cc = sum(1 for m in _D["clue"] if m) / len(pool_words)
    print(f"  in WordNet: board {bc:.1%}   clues {cc:.1%}", flush=True)

    n = len(pool_words)
    step = max(1, n // (args.workers * 4))
    chunks = [(i, min(i + step, n)) for i in range(0, n, step)]
    wup = np.full((n, len(board_words)), np.nan, dtype=np.float32)
    lcs = np.full((n, len(board_words)), np.nan, dtype=np.float32)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for lo, w, d in ex.map(score_chunk, chunks):
            wup[lo : lo + len(w)] = w
            lcs[lo : lo + len(d)] = d
            done += len(w)
            print(f"  {done:,}/{n:,}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, wup=wup, lcs_depth=lcs,
                        clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board_words))
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)   "
          f"nonzero {np.mean(np.nan_to_num(wup) > 0):.1%}")


if __name__ == "__main__":
    main()
