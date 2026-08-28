"""Generate training examples for M8's scorer (SCOPE.md §M7).

Each example is (features, outcome, reward) for one (sampled board state,
sampled clue, sampled training-pool guesser) triple:

- **features**: `build_features(board, clue, sims, turn_index)` -- §2's
  feature vector (see codenames/features.py).
- **outcome**: `codenames.scorer.outcome_class(k, cause)`, packing two
  things about this guesser's rollout into one training label: `k`, how
  many own-words it revealed before stopping (capped at MAX_K, matching
  §2's k in 0..4), and `cause`, which role (neutral/opponent/assassin)
  actually stopped it -- `None` iff k==MAX_K (hit the cap clean, no miss).
  Computed as a side-effect-free rollout over the guesser's own ranking --
  it peeks at `board.role_of()` and never calls `board.reveal()`, so many
  (clue, guesser) pairs can be evaluated against the exact same sampled
  board state without needing a fresh copy per pair. See
  `simulate_natural_stop`. Recording the cause (not just k) is what lets
  `codenames.scorer.reward_matrix` charge a neutral miss, an opponent
  miss, and an assassin miss differently instead of one flattened
  worst-case penalty -- see scorer.py's module docstring.
- **reward**: the reward SCOPE §2 attaches to that same rollout (own words
  +1 each, plus whatever ended the run -- another miss's reward, or nothing
  if it hit the k cap without a miss). Diagnostic only -- not read by
  scripts/train_scorer.py, which trains against `outcome` alone.

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
shard_size x feature_dim), `outcome_NNNNN.npy` (int32, one of
`codenames.scorer.N_OUTCOME_CLASSES` classes), `reward_NNNNN.npy`
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
from typing import Callable

import numpy as np
import torch

from codenames.board import MAX_CLUE_NUMBER as MAX_K
from codenames.board import Board, Role, is_legal_clue, load_training_wordlist
from codenames.clue_search import mean_from_columns, top_k_legal_clues
from codenames.features import build_features, feature_dim
from codenames.game import ROLE_REWARD
from codenames.gpu_clue_search import batched_mean_similarity
from codenames.guessers.base import Guesser
from codenames.guessers.registry import DEFAULT_POOL_CONFIG, training_pool
from codenames.scorer import outcome_class
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

# How many examples' worth of (board, clue-plan) to accumulate before one
# batched_mean_similarity call, when GPU batching is active. Independent
# of that function's own internal chunk_size (which bounds GPU memory per
# call) -- this just bounds how much board/plan state is held in Python
# lists at once between GPU calls. Large enough to amortize per-call
# overhead across many chunks, small enough to stay a trivial amount of
# host memory (a few thousand Board objects + word lists).
PLAN_BATCH_SIZE = 4096

CLUE_MIX = {"subset_topk": 0.6, "any_word_topk": 0.3, "random": 0.1}
TOPK_POOL = 20  # sample among the top-K neighbors, not always the single best
_RANDOM_CLUE_ATTEMPTS = 1000

# The board vocabulary is fixed at ~400 words, so across millions of sampled
# examples the *same* board word gets drawn over and over. Reading a word's
# full (n_clues, n_spaces) column back off the mmapped tensor is the
# dominant per-example cost on the CPU path -- caching it per word turns
# that into a one-time cost per distinct word actually sampled (at most
# ~400 columns, a few hundred MB), not per example. Kept local to this
# script rather than in clue_search.py: the arena runs many worker
# processes and already had one cache-related RSS blowup fixed (see
# codemasters/linear_scorer.py) -- a shared cache there would reintroduce
# the same risk for CentroidCodemaster. This script is a single process,
# so no such multiplication applies. Only used by the CPU fallback path
# now (sample_clue) -- the default GPU-batched path
# (codenames.gpu_clue_search) doesn't need per-word caching at all, since
# it reads straight off the GPU-resident tensor for a whole batch at once.
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


def _plan_clue(rng: random.Random, sims: SimilarityTensor, board: Board) -> tuple[str, list[str] | str] | None:
    """The RNG-consuming part of clue sampling, split out from actually
    scoring/picking a clue so a caller can batch the (expensive, full-
    vocabulary) scoring step across many boards at once -- see
    generate()'s GPU-batched path below. Returns `("scored", query_words)`
    for the two CLUE_MIX branches that need a mean-similarity score,
    `("resolved", clue)` for the "random legal clue" branch (already fully
    decided, nothing to score), or `None` on the rare occasions no
    candidate could even be planned (e.g. no unrevealed words left in the
    sampled bucket) -- callers should resample a fresh board."""
    roll = rng.random()

    if roll < CLUE_MIX["subset_topk"]:
        own_unrevealed = board.words_by_role(Role.OWN, unrevealed_only=True)
        if not own_unrevealed:
            return None
        subset_size = rng.randint(1, min(MAX_K, len(own_unrevealed)))
        return "scored", rng.sample(own_unrevealed, k=subset_size)
    elif roll < CLUE_MIX["subset_topk"] + CLUE_MIX["any_word_topk"]:
        unrevealed = [w for w in board.words if not board.is_revealed(w)]
        if not unrevealed:
            return None
        return "scored", [rng.choice(unrevealed)]
    else:
        for _ in range(_RANDOM_CLUE_ATTEMPTS):
            clue = rng.choice(sims.clue_words)
            if is_legal_clue(clue, board.words):
                return "resolved", clue
        return None


def _resolve_scored_clue(rng: random.Random, sims: SimilarityTensor, board: Board, scores: np.ndarray) -> str | None:
    candidates = top_k_legal_clues(sims, board, scores, k=TOPK_POOL)
    return rng.choice(candidates) if candidates else None


def sample_clue(rng: random.Random, sims: SimilarityTensor, board: Board) -> str | None:
    """A legal clue drawn per CLUE_MIX. Returns None on the rare occasions
    no legal candidate could be found -- callers should just resample.
    Scores one board at a time on CPU; generate()'s GPU-batched path
    doesn't call this directly (it batches _plan_clue/_resolve_scored_clue
    across many boards instead), but this stays as the single-example
    entrypoint other callers (and its own tests) use."""
    plan = _plan_clue(rng, sims, board)
    if plan is None:
        return None
    mode, payload = plan
    if mode == "resolved":
        return payload
    scores = _cached_mean_similarity_to_words(sims, payload)
    return _resolve_scored_clue(rng, sims, board, scores)


def simulate_natural_stop(
    board: Board, clue: str, guesser: Guesser, sims: SimilarityTensor, max_k: int = MAX_K
) -> tuple[int, Role | None, float]:
    """How many own-words `guesser` would reveal for `clue` before
    stopping (capped at max_k), which role stopped it (None iff the cap
    was hit with no miss), and the reward that rollout earns. Reads board
    state (`role_of`) but never mutates it (`reveal`) -- lets many (clue,
    guesser) pairs be evaluated against the same sampled board.

    Assumes `ranked` always contains a non-own word by the time k would
    otherwise reach max_k or `ranked` runs out -- true for this project's
    board sampling (the assassin is never revealed by
    `sample_partial_board`, so it's always an unrevealed candidate) and
    guesser pool (none of them truncate their own ranking). If that
    invariant is ever violated, a k<max_k result with no miss encountered
    falls through to `(k, None, reward)`, which `codenames.scorer.
    outcome_class` will reject rather than silently mislabel -- fail loud,
    not silently wrong."""
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
                return k, None, reward
        else:
            reward += ROLE_REWARD[role]
            return k, role, reward
    return k, None, reward


def _existing_shard_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("features_*.npy")))


def _plan_batch(
    rng: random.Random, sims: SimilarityTensor, board_vocabulary: list[str], target: int
) -> tuple[list[tuple[Board, int, str]], list[Board], list[int], list[list[str]]]:
    """Sample boards and plan clues (RNG-consuming, cheap) until `target`
    plans are collected. Returns (already-resolved (board, turn_index,
    clue) triples from the "random legal clue" branch, plus the three
    parallel lists -- boards/turn_indices/query_words -- for the branches
    that still need a mean-similarity score computed)."""
    resolved: list[tuple[Board, int, str]] = []
    pending_boards: list[Board] = []
    pending_turn_indices: list[int] = []
    pending_queries: list[list[str]] = []
    while len(resolved) + len(pending_boards) < target:
        board, turn_index = sample_partial_board(rng, board_vocabulary)
        plan = _plan_clue(rng, sims, board)
        if plan is None:
            continue
        mode, payload = plan
        if mode == "resolved":
            resolved.append((board, turn_index, payload))
        else:
            pending_boards.append(board)
            pending_turn_indices.append(turn_index)
            pending_queries.append(payload)
    return resolved, pending_boards, pending_turn_indices, pending_queries


def generate(
    n_examples: int,
    shard_size: int,
    output_dir: Path,
    seed: int,
    guesser_pool_config: Path = DEFAULT_POOL_CONFIG,
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    board_vocabulary: list[str] | None = None,
    feature_builder: Callable[[Board, str, SimilarityTensor, int], np.ndarray] = build_features,
    guesser_weights: dict[str, float] | None = None,
    use_gpu_batch: bool = True,
) -> int:
    """`feature_builder` and `guesser_weights` exist for SCOPE §9's
    ablations (scripts/run_ablation_study.py), not as CLI flags: swapping
    in `build_features_unsorted` reproduces the exact same sampled
    boards/clues/guessers for a given `seed` (feature computation never
    consumes randomness), and `guesser_weights` skews which guesser labels
    each example without changing anything else about the sampling.

    `use_gpu_batch` (on by default, falls back to CPU automatically
    without CUDA): clue *scoring* -- the dominant per-example cost,
    measured ~96% of it, see docs/log.md's GPU-data-generation entries --
    runs batched across PLAN_BATCH_SIZE examples at once via
    codenames.gpu_clue_search instead of one board at a time. This
    reorders the RNG draw sequence relative to the old one-example-at-a-
    time loop (planning for many boards happens before any of their clues
    get resolved or their guesser gets picked), so a given seed's exact
    shard contents differ from what an older version of this function
    produced -- still fully deterministic for a *given* version of this
    function, which is what scripts/run_ablation_study.py's same-seed
    reuse across feature_builder/guesser_weights variants actually
    depends on, not byte-for-byte stability across code changes."""
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
    guesser_choice_weights = [guesser_weights[name] for name in guesser_names] if guesser_weights is not None else None

    device = torch.device("cuda") if (use_gpu_batch and torch.cuda.is_available()) else None

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = _existing_shard_count(output_dir)

    dim = feature_dim(len(sims.spaces))
    produced = 0
    start_time = time.time()

    def emit(features, outcomes, rewards, seeds, filled: int, board: Board, turn_index: int, clue: str) -> int:
        if guesser_choice_weights is not None:
            guesser_name = rng.choices(guesser_names, weights=guesser_choice_weights, k=1)[0]
        else:
            guesser_name = rng.choice(guesser_names)
        guesser = guessers[guesser_name]
        features[filled] = feature_builder(board, clue, sims, turn_index)
        k, cause, rewards[filled] = simulate_natural_stop(board, clue, guesser, sims)
        outcomes[filled] = outcome_class(k, cause)
        seeds[filled] = board.seed
        return filled + 1

    while produced < n_examples:
        this_shard_size = min(shard_size, n_examples - produced)
        features = np.empty((this_shard_size, dim), dtype=np.float32)
        outcomes = np.empty(this_shard_size, dtype=np.int32)
        rewards = np.empty(this_shard_size, dtype=np.float32)
        seeds = np.empty(this_shard_size, dtype=np.int64)

        filled = 0
        while filled < this_shard_size:
            if device is None:
                board, turn_index = sample_partial_board(rng, board_vocabulary)
                clue = sample_clue(rng, sims, board)
                if clue is None:
                    continue
                filled = emit(features, outcomes, rewards, seeds, filled, board, turn_index, clue)
                produced += 1
                continue

            target = min(PLAN_BATCH_SIZE, this_shard_size - filled)
            resolved, pending_boards, pending_turn_indices, pending_queries = _plan_batch(rng, sims, board_vocabulary, target)
            if pending_queries:
                scores_batch = batched_mean_similarity(sims, pending_queries, device)
                for board, turn_index, scores in zip(pending_boards, pending_turn_indices, scores_batch):
                    clue = _resolve_scored_clue(rng, sims, board, scores)
                    if clue is not None:
                        resolved.append((board, turn_index, clue))

            for board, turn_index, clue in resolved:
                if filled >= this_shard_size:
                    break
                filled = emit(features, outcomes, rewards, seeds, filled, board, turn_index, clue)
                produced += 1

        np.save(output_dir / f"features_{shard_index:05d}.npy", features)
        np.save(output_dir / f"outcome_{shard_index:05d}.npy", outcomes)
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
    parser.add_argument(
        "--no-gpu-batch", action="store_true", help="score clues one board at a time on CPU instead of batched on GPU (default: batched GPU, ~30x faster on the dominant cost, see docs/log.md)"
    )
    args = parser.parse_args()

    produced = generate(
        n_examples=args.n_examples,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        seed=args.seed,
        guesser_pool_config=args.guesser_pool_config,
        use_gpu_batch=not args.no_gpu_batch,
    )
    print(f"done: {produced} examples written to {args.output_dir}")


if __name__ == "__main__":
    main()
