"""Precompute human-association strengths from clue words to board words.

SWOW (De Deyne et al. 2019) records what word people say when cued with
another word: 1.39M cue->response pairs over 12,217 cues, from ~90k
participants. That is the listener's task measured directly on humans, and it
is a different kind of evidence from an embedding -- embeddings measure
*similarity* (words used in similar contexts), association measures
*relatedness* (words that come to mind together). `nikon` and `Olympus` are
not similar; they are associated.

**Why two hops.** Association data is a sparse graph, not a dense space. Only
0.57 of a 25-word board has a direct SWOW entry from a given clue, so a
one-hop feature is empty ~98% of the time and cannot move any metric. Walking
two hops -- clue -> intermediate -> board word -- reaches 15.6 of 25, because
the intermediate layer is where the graph's density lives. Measured, not
assumed; the same walk on the smaller USF norms only reaches 8.2.

Paths are combined by sum of products rather than max: several weak routes
from a clue to a word is real evidence of relatedness, and summing accumulates
it where a max would discard all but one. That also makes the whole thing one
sparse matrix multiply.

Output is `cache/swow.npz`: two sparse (n_clue_pool x n_board_words) matrices,
one-hop and two-hop, aligned to the similarity tensor's own clue and board
indices so the feature extractor can look up a row without re-deriving
anything.

Data is CC BY-NC-ND 3.0 and lives under the gitignored data/ directory; only
this derived table is cached.

Usage:
    python scripts/data/build_swow_tables.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

DEFAULT_SWOW = PROJECT_ROOT / "data" / "free_association" / "swow" / "strength.SWOW-EN.R123.20180827.csv"
OUT = PROJECT_ROOT / "cache" / "swow.npz"


def load_strengths(path: Path) -> dict[str, dict[str, float]]:
    fwd: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            try:
                s = float(p[4])
            except ValueError:
                continue
            fwd[p[0].lower()][p[1].lower()] = s
    return fwd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swow", type=Path, default=DEFAULT_SWOW)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.swow.exists():
        raise SystemExit(f"SWOW strength file not found at {args.swow}\n"
                         "Download SWOW-EN18 from https://smallworldofwords.org/en/project/research "
                         "and unzip strength.SWOW-EN.R123.*.csv into data/free_association/swow/")

    print(f"reading {args.swow.name}...", flush=True)
    fwd = load_strengths(args.swow)
    print(f"  {sum(len(v) for v in fwd.values()):,} pairs over {len(fwd):,} cues")

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    clue_words = [w.lower() for w in stats.clue_words]
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    board_pos = {w: i for i, w in enumerate(board_words)}

    # Intermediate layer: every word that is either a SWOW cue or a response.
    # Restricting it to board words would defeat the point -- the density that
    # makes two hops work lives in words that are not on any board.
    mid = sorted({w for w in fwd} | {r for d in fwd.values() for r in d})
    mid_pos = {w: i for i, w in enumerate(mid)}
    print(f"  intermediate vocabulary: {len(mid):,}")

    # A: clue -> intermediate.  B: intermediate -> board word.
    ai, aj, av = [], [], []
    for ci, c in enumerate(clue_words):
        for r, s in fwd.get(c, {}).items():
            ai.append(ci); aj.append(mid_pos[r]); av.append(s)
    A = sp.csr_matrix((av, (ai, aj)), shape=(len(clue_words), len(mid)), dtype=np.float32)

    bi, bj, bv = [], [], []
    for w, d in fwd.items():
        wi = mid_pos[w]
        for r, s in d.items():
            j = board_pos.get(r)
            if j is not None:
                bi.append(wi); bj.append(j); bv.append(s)
    B = sp.csr_matrix((bv, (bi, bj)), shape=(len(mid), len(board_words)), dtype=np.float32)
    print(f"  A nnz {A.nnz:,}   B nnz {B.nnz:,}")

    # One hop is A restricted to the board columns.
    keep = np.array([mid_pos[w] for w in board_words if w in mid_pos])
    cols = np.array([board_pos[w] for w in board_words if w in mid_pos])
    one = sp.csr_matrix((len(clue_words), len(board_words)), dtype=np.float32)
    sel = sp.csr_matrix((np.ones(len(keep), dtype=np.float32), (keep, cols)),
                        shape=(len(mid), len(board_words)))
    one = (A @ sel).tocsr()

    print("multiplying for two-hop...", flush=True)
    two = (A @ B).tocsr()
    print(f"  one-hop nnz {one.nnz:,}   two-hop nnz {two.nnz:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        one_data=one.data, one_indices=one.indices, one_indptr=one.indptr, one_shape=one.shape,
        two_data=two.data, two_indices=two.indices, two_indptr=two.indptr, two_shape=two.shape,
        board_words=np.array(board_words),
    )
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
