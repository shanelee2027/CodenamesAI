"""Generate training examples for M8's scorer (SCOPE.md §M7).

Each example is (features, k, reward) for one (sampled board state, sampled
clue, sampled training-pool guesser) triple:

- **features**: `build_features(board, clue, sims, turn_index)` -- §2's
  feature vector (see codenames/features.py).
- **k**: how many own-words this guesser would reveal for this clue before
  stopping, capped at MAX_K (matches §2's k in 0..4). Computed as a
  side-effect-free rollout over the guesser's own ranking -- it peeks at
  `board.role_of()` and never calls `board.reveal()`, so many (clue,
  guesser) pairs can be evaluated against the exact same sampled board
  state without needing a fresh copy per pair. See `simulate_natural_stop`.
- **reward**: the reward SCOPE §2 attaches to that same rollout (own words
  +1 each, plus whatever ended the run -- another miss's reward, or nothing
  if it hit the k cap without a miss).

Guesser sampling still goes through `training_pool()` (SCOPE §3/§5's
mechanism for "training code must never touch held-out guessers"), though
the first-pass pool (configs/guesser_pool.json) currently has none held out
-- see docs/log.md's post-M8 design-revision entry for why.

**Clue sampling mix** (SCOPE §5 M7): ~60% top-k neighbors of a random
own-word subset, ~30% top-k neighbors of one random board word of *any*
role (this is where dangerous assassin-adjacent clues come from -- the
model must see them), ~10% uniformly random legal clues. "Top-k neighbors"
means: sample among the K best-scoring legal clues, not always the single
best, since always taking the argmax would make 60%+30% of the dataset
degenerate to a handful of the objectively-closest words per board.

**Board sampling** includes partially-revealed states (SCOPE §5 M7): each
sampled board reveals a random number of own/opponent/neutral words (never
the assassin, and always leaving >=1 own word unrevealed -- a state where
the game has already ended isn't a useful training example). `turn_index`
is approximated as the total number of revealed words at sampling time,
since these are synthetic states, not the output of an actually-played
game. Boards are sampled from `load_training_wordlist()` (the full board
vocabulary minus the held-out subset, board.py) by default -- generalization
to boards built from never-seen words is checked separately, not trained
against here (first-pass revision, see docs/log.md: this replaces held-out
guessers as the generalization check).

**Output** is sharded .npy files under `--output-dir` (default
cache/training_data/, gitignored): `features_NNNNN.npy` (float32,
shard_size x feature_dim), `k_NNNNN.npy` (int32), `reward_NNNNN.npy`
(float32), `seed_NNNNN.npy` (int64, the sampled board's seed) -- each
independently mmap-loadable. `seed` exists specifically so M8's training
script can split by board seed rather than by row (SCOPE §4: "the same
board appears in many training examples [if reused]; row-wise splits leak
boards across train/val"). This is the concrete meaning of "appendable
mmapped output" here: re-running this script adds new shards after
whatever already exists in the output directory, rather than needing to
know the eventual total size up front or resizing an existing file.

Usage:
    python scripts/generate_training_data.py --n-examples 100000
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from codenames.board import MAX_CLUE_NUMBER as MAX_K
from codenames.board import Board, Role, is_legal_clue, load_training_wordlist
from codenames.clue_search import mean_from_columns, top_k_legal_clues
from codenames.features import build_features, feature_dim
from codenames.game import ROLE_REWARD
from codenames.guessers.base import Guesser
from codenames.guessers.registry import DEFAULT_POOL_CONFIG, training_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

CLUE_MIX = {"subset_topk": 0.6, "any_word_topk": 0.3, "random": 0.1}
TOPK_POOL = 20  # sample among the top-K neighbors, not always the single best
_RANDOM_CLUE_ATTEMPTS = 1000

# The board vocabulary is fixed at ~400 words, so across millions of sampled
# examples the *same* board word gets drawn over and over. Reading a word's
# full (n_clues, n_spaces) column back off the mmapped tensor is the
# dominant per-example cost (measured ~40ms/example, >95% of it here) --
# caching it per word turns that into a one-time cost per distinct word
# actually sampled (at most ~400 columns, a few hundred MB), not per
# example. Kept local to this script rather than in clue_search.py: the
# arena runs many worker processes and already had one cache-related RSS
# blowup fixed (see codemasters/linear_scorer.py) -- a shared cache there
# would reintroduce the same risk for CentroidCodemaster. This script is a
# single process, so no such multiplication applies.
_column_cache: dict[str, np.ndarray] = {}


def _cached_column(sims: SimilarityTensor, word: str) -> np.ndarray:
    key = word.lower()
    column = _column_cache.get(key)
    if column is None:
        column = np.asarray(sims.tensor[:, sims.board_index[key], :], dtype=np.float32)
        _column_cache[key] = column
    return column


def _cached_mean_similarity_to_words(sims: SimilarityTensor, words: list[str]) -> np.ndarray:
    return mean_from_columns([_cached_column(sims, w) for w in words])


def sample_partial_board(rng: random.Random, vocabulary: list[str]) -> tuple[Board, int]:
    """A board with a random number of own/opponent/neutral words already
    revealed (never the assassin; always >=1 own word left unrevealed).
    Returns (board, revealed_count) -- revealed_count stands in for
    `turn_index` in build_features. `vocabulary` should be
    `load_training_wordlist()` (the default in `generate()`) so training
    data never includes a held-out board word -- see board.py."""
    seed = rng.randrange(2**31)
    board = Board.generate(seed=seed, vocabulary=vocabulary)
    revealed_count = 0
    for role in (Role.OWN, Role.OPPONENT, Role.NEUTRAL):
        words = board.words_by_role(role)
        max_reveal = len(words) - 1 if role == Role.OWN else len(words)
        n_reveal = rng.randint(0, max_reveal)
        for w in rng.sample(words, k=n_reveal):
            board.reveal(w)
        revealed_count += n_reveal
    return board, revealed_count


def sample_clue(rng: random.Random, sims: SimilarityTensor, board: Board) -> str | None:
    """A legal clue drawn per CLUE_MIX. Returns None on the rare occasions
    no legal candidate could be found (e.g. no unrevealed words left in
    the sampled bucket) -- callers should just resample."""
    roll = rng.random()

    if roll < CLUE_MIX["subset_topk"]:
        own_unrevealed = board.words_by_role(Role.OWN, unrevealed_only=True)
        if not own_unrevealed:
            return None
        subset_size = rng.randint(1, min(MAX_K, len(own_unrevealed)))
        subset = rng.sample(own_unrevealed, k=subset_size)
        scores = _cached_mean_similarity_to_words(sims, subset)
    elif roll < CLUE_MIX["subset_topk"] + CLUE_MIX["any_word_topk"]:
        unrevealed = [w for w in board.words if not board.is_revealed(w)]
        if not unrevealed:
            return None
        scores = _cached_mean_similarity_to_words(sims, [rng.choice(unrevealed)])
    else:
        for _ in range(_RANDOM_CLUE_ATTEMPTS):
            clue = rng.choice(sims.clue_words)
            if is_legal_clue(clue, board.words):
                return clue
        return None

    candidates = top_k_legal_clues(sims, board, scores, k=TOPK_POOL)
    return rng.choice(candidates) if candidates else None


def simulate_natural_stop(board: Board, clue: str, guesser: Guesser, sims: SimilarityTensor, max_k: int = MAX_K) -> tuple[int, float]:
    """How many own-words `guesser` would reveal for `clue` before
    stopping (capped at max_k), and the reward that rollout earns. Reads
    board state (`role_of`) but never mutates it (`reveal`) -- lets many
    (clue, guesser) pairs be evaluated against the same sampled board."""
    candidates = [w for w in board.words if not board.is_revealed(w)]
    ranked = guesser.rank_candidates(clue, candidates, sims)

    k = 0
    reward = 0.0
    for word in ranked:
        role = board.role_of(word)
        if role == Role.OWN:
            k += 1
            reward += ROLE_REWARD[Role.OWN]
            if k >= max_k:
                break
        else:
            reward += ROLE_REWARD[role]
            break
    return k, reward


def _existing_shard_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("features_*.npy")))


def generate(
    n_examples: int,
    shard_size: int,
    output_dir: Path,
    seed: int,
    guesser_pool_config: Path = DEFAULT_POOL_CONFIG,
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    board_vocabulary: list[str] | None = None,
) -> int:
    # Hardcoded default, not a CLI flag: training data must never include a
    # held-out board word by construction, the same way held-out guessers
    # used to be enforced by training_pool() rather than by convention.
    if board_vocabulary is None:
        board_vocabulary = load_training_wordlist()

    sims = SimilarityTensor.load(sims_cache_dir)
    guessers = training_pool(guesser_pool_config)
    guesser_names = list(guessers.keys())
    if not guesser_names:
        raise ValueError(f"training_pool({guesser_pool_config}) is empty -- nothing to sample from")

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = _existing_shard_count(output_dir)

    dim = feature_dim(len(sims.spaces))
    produced = 0
    start_time = time.time()

    while produced < n_examples:
        this_shard_size = min(shard_size, n_examples - produced)
        features = np.empty((this_shard_size, dim), dtype=np.float32)
        ks = np.empty(this_shard_size, dtype=np.int32)
        rewards = np.empty(this_shard_size, dtype=np.float32)
        seeds = np.empty(this_shard_size, dtype=np.int64)

        filled = 0
        while filled < this_shard_size:
            board, turn_index = sample_partial_board(rng, board_vocabulary)
            clue = sample_clue(rng, sims, board)
            if clue is None:
                continue

            guesser = guessers[rng.choice(guesser_names)]
            features[filled] = build_features(board, clue, sims, turn_index)
            ks[filled], rewards[filled] = simulate_natural_stop(board, clue, guesser, sims)
            seeds[filled] = board.seed
            filled += 1
            produced += 1

        np.save(output_dir / f"features_{shard_index:05d}.npy", features)
        np.save(output_dir / f"k_{shard_index:05d}.npy", ks)
        np.save(output_dir / f"reward_{shard_index:05d}.npy", rewards)
        np.save(output_dir / f"seed_{shard_index:05d}.npy", seeds)

        elapsed = time.time() - start_time
        rate = produced / elapsed if elapsed > 0 else 0.0
        print(f"shard {shard_index:05d}: +{filled} examples ({produced}/{n_examples} total, {rate:.0f} examples/sec)")
        shard_index += 1

    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-examples", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/training_data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    args = parser.parse_args()

    produced = generate(
        n_examples=args.n_examples,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        seed=args.seed,
        guesser_pool_config=args.guesser_pool_config,
    )
    print(f"done: {produced} examples written to {args.output_dir}")


if __name__ == "__main__":
    main()
