"""Given a clue, rank board words by similarity in each embedding space.

The reverse direction of scripts/sanity_check_sims.py (which takes a board
word and finds top clues): this takes a clue -- any word in the 250k-word
clue vocabulary, not restricted to the 400 board words -- and ranks board
words by similarity to it. Useful for manually testing whether a candidate
clue would actually point at the board words you'd expect.

Usage:
    python scripts/check_clue.py --clue technoblade
    python scripts/check_clue.py --clue king --board-words King Queen Egypt Shark
    python scripts/check_clue.py --clue spy --top 15
"""

from __future__ import annotations

import argparse

import numpy as np

from codenames.similarity import SimilarityTensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clue", required=True, help="a word from the clue vocabulary (not necessarily a board word)")
    parser.add_argument("--board-words", nargs="*", default=None, help="restrict to these board words (default: all 400)")
    parser.add_argument("--top", type=int, default=None, help="show only the top N board words (ranked by mean similarity across spaces); default: show all")
    args = parser.parse_args()

    sims = SimilarityTensor.load()
    board_words = args.board_words if args.board_words is not None else sims.board_words

    if args.clue.lower() not in sims.clue_index:
        raise SystemExit(
            f"{args.clue!r} is not in the clue vocabulary (250k lowercase alphabetic "
            "GloVe-ranked words) -- check spelling, or it may be too rare/not purely "
            "alphabetic to have made the cutoff."
        )

    try:
        result = sims.similarities_for_board(args.clue, board_words)  # (n_board, n_spaces)
    except KeyError as e:
        raise SystemExit(str(e))

    # rank by mean of the valid (non-NaN) space values, so a word missing
    # from one space doesn't just silently sort as worst-possible
    order = np.argsort(-np.nanmean(result, axis=1))
    if args.top is not None:
        order = order[: args.top]

    header = f"{'board word':20s} " + " ".join(f"{s:>14s}" for s in sims.spaces)
    print(f"clue: {args.clue}\n")
    print(header)
    for i in order:
        row = " ".join(
            "           n/a" if np.isnan(v) else f"{v:14.4f}"
            for v in result[i]
        )
        print(f"{board_words[i]:20s} {row}")


if __name__ == "__main__":
    main()
