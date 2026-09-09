"""Turn a stored rollout set into a training dataset
(docs/iteration-architecture.md step 4).

This is the cheap, model-specific half of what
`scripts/pipeline/generate_training_data.py` used to do in one pass. That script now
writes rollouts -- (board state, clue, guesser) -> (k, cause), the expensive
model-*independent* part, see `codenames/rollouts.py` -- and this one applies
a feature builder to them.

The point of the split: the feature vector is expected to change between
models, and re-simulating identical rollouts to get different columns out of
them is the dominant cost in the pipeline (`scripts/pipeline/run_ablation_study.py`
measures generation at ~40 min at moderate scale, against fast training).
Featurizing a stored rollout set is a fraction of that, so a new feature
design is a re-featurization rather than a regeneration -- and every model
trains on the *identical* rollouts, which also makes model-to-model
comparisons cleaner.

**Output is byte-compatible with the old dataset layout on purpose**:
`features_NNNNN.npy`, `outcome_NNNNN.npy`, `reward_NNNNN.npy`,
`seed_NNNNN.npy`, exactly as `scripts/pipeline/train_scorer.py` already reads them.
That script needed no changes at all, and a dataset produced from rollouts
is interchangeable with one produced by the pre-split pipeline -- which is
what makes the equivalence check in docs/log.md possible.

`reward` is recomputed here from (k, cause) via
`codenames.rollouts.reward_for` rather than read from storage, so changing a
reward constant reprices an existing rollout set for free. `outcome` is
recomputed via `codenames.scorer.outcome_class` for the same reason.

Usage:
    python scripts/pipeline/featurize_rollouts.py --rollout-dir cache/rollouts \\
        --output-dir cache/training_data
    python scripts/pipeline/featurize_rollouts.py --feature-builder unsorted ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/pipeline/<this> -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board  # noqa: E402
from codenames.features import build_features, build_features_unsorted  # noqa: E402
from codenames.game import ROLE_REWARD  # noqa: E402
from codenames.rollouts import (  # noqa: E402
    CAUSE_NONE,
    ROLE_BY_CODE,
    board_from_row,
    clue_vocab_fingerprint,
    load_manifest,
    load_rollouts,
    reward_for,
    shard_indices,
)
from codenames.scorer import outcome_class  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

# Named builders so a variant is a CLI flag rather than an import edit. The
# ablations in scripts/pipeline/run_ablation_study.py that used to need their own
# generation pass (`unsorted`) are now just another entry here over the same
# stored rollouts.
FEATURE_BUILDERS: dict[str, Callable[[Board, str, SimilarityTensor, int], np.ndarray]] = {
    "sorted": build_features,
    "unsorted": build_features_unsorted,
}


def featurize(
    rollout_dir: Path,
    output_dir: Path,
    feature_builder: Callable[[Board, str, SimilarityTensor, int], np.ndarray] = build_features,
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
) -> int:
    """Featurize every shard of `rollout_dir` into `output_dir`. Returns
    the number of examples written. One output shard per input shard, so
    the sharding (and its mmap-ability) carries straight through."""
    manifest = load_manifest(rollout_dir)
    sims = SimilarityTensor.load(sims_cache_dir)

    # Clue indices are meaningless against a different clue vocabulary, and a
    # silent mismatch would mislabel every row rather than fail. Same reason
    # codenames/similarity.py validates its own cached metadata.
    actual = clue_vocab_fingerprint(sims.clue_words)
    if actual != manifest["clue_vocab_hash"]:
        raise ValueError(
            f"rollout set {rollout_dir} was written against clue vocabulary "
            f"{manifest['clue_vocab_hash']} ({manifest['n_clue_words']} words), but "
            f"{sims_cache_dir} holds {actual} ({len(sims.clue_words)} words) -- "
            "re-generate the rollouts or point at the matching similarity tensor"
        )

    board_vocab = manifest["board_vocab"]
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    start = time.time()
    for shard in shard_indices(rollout_dir):
        batch = load_rollouts(rollout_dir, shard)
        n = len(batch)

        features = None
        outcomes = np.empty(n, dtype=np.int32)
        rewards = np.empty(n, dtype=np.float32)
        seeds = np.empty(n, dtype=np.int64)

        for i in range(n):
            board = board_from_row(batch, i, board_vocab)
            clue = sims.clue_words[int(batch.clue[i])]
            row = feature_builder(board, clue, sims, int(batch.turn_index[i]))
            if features is None:
                # Width comes from the builder, not from a constant, so a
                # feature design that changes the vector's size needs no
                # change here.
                features = np.empty((n, row.shape[0]), dtype=np.float32)
            features[i] = row

            k = int(batch.k[i])
            cause_code = int(batch.cause[i])
            cause = None if cause_code == CAUSE_NONE else ROLE_BY_CODE[cause_code]
            outcomes[i] = outcome_class(k, cause)
            rewards[i] = reward_for(k, cause, ROLE_REWARD)
            seeds[i] = int(batch.board_seed[i])

        np.save(output_dir / f"features_{shard:05d}.npy", features)
        np.save(output_dir / f"outcome_{shard:05d}.npy", outcomes)
        np.save(output_dir / f"reward_{shard:05d}.npy", rewards)
        np.save(output_dir / f"seed_{shard:05d}.npy", seeds)

        total += n
        rate = total / (time.time() - start) if time.time() > start else 0.0
        print(f"shard {shard:05d}: {n} examples ({total} total, {rate:.0f} examples/sec)")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout-dir", type=Path, default=Path("cache/rollouts"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/training_data"))
    parser.add_argument(
        "--feature-builder",
        choices=sorted(FEATURE_BUILDERS),
        default="sorted",
        help="which feature vector to build from the stored rollouts (default: sorted)",
    )
    parser.add_argument("--sims-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    written = featurize(
        rollout_dir=args.rollout_dir,
        output_dir=args.output_dir,
        feature_builder=FEATURE_BUILDERS[args.feature_builder],
        sims_cache_dir=args.sims_cache_dir,
    )
    print(f"done: {written} examples written to {args.output_dir}")


if __name__ == "__main__":
    main()
