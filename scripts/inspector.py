"""The inspector (SCOPE.md §M3): given a board and a typed clue, show what
every embedding space thinks, whether the clue is legal, what each guesser
would pick, and a baseline score. This tool is meant to stay useful for
the entire project -- it's where you go to sanity-check "why did the
model like/dislike this clue?"

One piece of what SCOPE.md asks for is still a placeholder, since this
was built out of milestone order (M3 before M8 per SCOPE, not after):
"a baseline score" is shown as an UNTUNED preview of SCOPE.md §6's own
baseline-3 formula (own +1, opponent -1, neutral -0.3, assassin -10,
averaged unweighted across the available spaces) -- not the real
baseline, which needs CMA-ES tuning against the guesser pool (M8).
Labeled as such in the output.

"What each guesser would pick" only shows each guesser's own preference
ranking, not a full simulated turn -- the number attempt cap and
turn-ending-on-a-miss rule are game-loop concerns (M6), not built yet.

Usage:
    python scripts/inspector.py --seed 42 --clue king
    python scripts/inspector.py --seed 42 --clue king --reveal Shark Egypt
"""

from __future__ import annotations

import argparse

import numpy as np

from codenames.board import Board, Role, is_legal_clue
from codenames.guessers import load_pool
from codenames.similarity import SimilarityTensor

BASELINE_ROLE_WEIGHTS = {
    Role.OWN: 1.0,
    Role.OPPONENT: -1.0,
    Role.NEUTRAL: -0.3,
    Role.ASSASSIN: -10.0,
}

ROLE_LABELS = {
    Role.OWN: "OWN",
    Role.OPPONENT: "OPPONENT",
    Role.NEUTRAL: "neutral",
    Role.ASSASSIN: "ASSASSIN",
}


def fmt(value: float) -> str:
    return "   n/a" if np.isnan(value) else f"{value:6.3f}"


def print_full_table(sims: SimilarityTensor, board: Board, clue: str) -> None:
    print(f"{'word':16s} {'role':10s} {'revealed':9s} " + " ".join(f"{s:>8s}" for s in sims.spaces))
    for role in Role:
        for word in board.words_by_role(role):
            values = sims.similarity(clue, word)
            revealed = "yes" if board.is_revealed(word) else ""
            row = " ".join(f"{v:8.3f}" if not np.isnan(v) else "     n/a" for v in values)
            print(f"{word:16s} {ROLE_LABELS[role]:10s} {revealed:9s} {row}")


def print_top_ranked_per_space(sims: SimilarityTensor, board: Board, clue: str, top: int) -> None:
    for space in sims.spaces:
        pairs = []
        for word in board.words:
            v = sims.similarity(clue, word, space=space)
            if not np.isnan(v):
                pairs.append((word, v, board.role_of(word)))
        pairs.sort(key=lambda p: -p[1])
        print(f"  --- {space} ---")
        for word, v, role in pairs[:top]:
            print(f"    {word:16s} {ROLE_LABELS[role]:10s} {v:.3f}")


def print_guesser_predictions(sims: SimilarityTensor, board: Board, clue: str, top: int) -> None:
    unrevealed = [w for w in board.words if not board.is_revealed(w)]
    for name, entry in load_pool().items():
        ranked = entry.guesser.rank_candidates(clue, unrevealed, sims)[:top]
        tag = "held-out" if entry.held_out else "training"
        if not ranked:
            picks = "(would guess nothing -- stopped immediately)"
        else:
            picks = ", ".join(f"{w} ({ROLE_LABELS[board.role_of(w)]})" for w in ranked)
        print(f"  {name:20s} ({tag:8s}): {picks}")


def baseline_score(sims: SimilarityTensor, board: Board, clue: str) -> tuple[float, dict[Role, float]]:
    """Untuned preview of SCOPE.md §6 baseline 3: weighted average across
    spaces, then weighted sum across roles. Real baseline 3 requires
    CMA-ES-tuned constants against the guesser pool (M6/M8) -- this uses
    SCOPE's stated example constants directly, unweighted across spaces."""
    role_means: dict[Role, float] = {}
    for role in Role:
        words = board.words_by_role(role, unrevealed_only=True)
        if not words:
            role_means[role] = 0.0
            continue
        per_word_means = []
        for w in words:
            values = sims.similarity(clue, w)
            valid = values[~np.isnan(values)]
            if len(valid):
                per_word_means.append(float(valid.mean()))
        role_means[role] = float(np.mean(per_word_means)) if per_word_means else 0.0

    total = sum(BASELINE_ROLE_WEIGHTS[role] * role_means[role] for role in Role)
    return total, role_means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="board seed (see codenames.board.Board.generate)")
    parser.add_argument("--clue", required=True)
    parser.add_argument("--reveal", nargs="*", default=[], help="board words to mark as already revealed")
    parser.add_argument("--top", type=int, default=25, help="how many board words to show in the per-space ranking (default: all 25)")
    parser.add_argument("--guesser-top", type=int, default=5, help="how many picks to show per guesser (default: 5, since a real turn only gets number attempts)")
    args = parser.parse_args()

    sims = SimilarityTensor.load()
    board = Board.generate(seed=args.seed)
    for word in args.reveal:
        board.reveal(word)

    print(f"board seed: {args.seed}")
    print(f"clue: {args.clue!r}\n")

    legal = is_legal_clue(args.clue, board.words)
    print(f"legal clue: {legal}" + ("" if legal else "  (contains or is contained by a board word)"))
    if args.clue.lower() not in sims.clue_index:
        print(f"\n{args.clue!r} is not in the clue vocabulary -- no similarity data available.")
        return
    print()

    print("=== per-space similarity to all 25 board words ===")
    print_full_table(sims, board, args.clue)

    print(f"\n=== per-space top-{args.top} board words ranked by similarity ===")
    print_top_ranked_per_space(sims, board, args.clue, args.top)

    print("\n=== what each guesser would pick (own preference ranking, not a full simulated turn) ===")
    print_guesser_predictions(sims, board, args.clue, args.guesser_top)

    print("\n=== baseline score (UNTUNED preview of SCOPE.md §6 baseline 3, not the real tuned baseline) ===")
    total, role_means = baseline_score(sims, board, args.clue)
    for role in Role:
        print(f"  {ROLE_LABELS[role]:10s} mean similarity (unrevealed): {role_means[role]:.3f}  x weight {BASELINE_ROLE_WEIGHTS[role]:+.1f}")
    print(f"  score: {total:.3f}")


if __name__ == "__main__":
    main()
