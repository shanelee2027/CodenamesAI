"""Print top-20 nearest clues for sample board words (SCOPE.md §M2/§M4).

Do not proceed past M2 (or after extending the tensor in M4) without
eyeballing this output. If the tensor indexing is wrong, everything
downstream is wrong and it will not be obvious from unit tests alone.

Prints one block per space found in the loaded tensor, so a newly added
space (via scripts/extend_similarity_tensor.py) gets checked automatically
without needing a --space flag.

Usage:
    python scripts/sanity_check_sims.py
    python scripts/sanity_check_sims.py --words king dream craft sword
"""

from __future__ import annotations

import argparse
import random

from codenames.similarity import SimilarityTensor

# A mix of common-noun and proper-noun board words, plus one multi-word
# entry (New york) to eyeball the mean-pooling path specifically. Board
# vocabulary lookups are case-insensitive, but these still need to be real
# entries in codenames/assets/board_words.txt.
DEFAULT_WORDS = ["King", "Bat", "New york", "Egypt", "Spy", "Shark"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", nargs="*", default=None, help="board words to check (default: a fixed sample + one random word)")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    sims = SimilarityTensor.load()
    print(f"loaded tensor {sims.tensor.shape}, spaces={sims.spaces}\n")

    words = args.words if args.words is not None else DEFAULT_WORDS + [random.choice(sims.board_words)]
    for word in words:
        print(f"=== {word} ===")
        for space in sims.spaces:
            try:
                top = sims.top_clues(word, k=args.k, space=space)
            except KeyError:
                print("  NOT IN BOARD VOCABULARY, skipping\n")
                break
            print(f"  --- {space} ---")
            for clue, score in top:
                print(f"    {clue:20s} {score:.4f}")
        print()


if __name__ == "__main__":
    main()
