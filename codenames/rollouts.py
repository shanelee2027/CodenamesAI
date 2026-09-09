"""Stored guesser rollouts -- the model-independent half of training data
(docs/iteration-architecture.md step 4).

`scripts/pipeline/generate_training_data.py` used to compute a feature vector and a
rollout outcome together and save only the *featurized* row, discarding
the board, clue, turn index and guesser. That welded the expensive,
model-independent half (simulating what a guesser does with a clue) to the
cheap, model-specific half (turning a board+clue into numbers). Since the
feature vector is expected to change between models, every new model would
otherwise re-simulate identical rollouts just to get different columns out
of them -- and simulation is the dominant cost (`scripts/pipeline/run_ablation_study.py`
measures ~40 min at moderate scale, against fast training).

So this module stores the rollout itself:

    (board state, clue, guesser, turn_index) -> (k, cause)

and features become a pure function applied on demand
(`scripts/pipeline/featurize_rollouts.py`). The pipeline is three layers rather
than two -- rollouts (shared, expensive) -> features (per model, cheap,
cached) -> training -- which is the same "compute the expensive
model-independent thing once" discipline the similarity tensor
(`codenames/similarity.py`) and the LLM response cache
(`codenames/llm_store.py`) already apply to the project's other two
expensive invariants.

**Reward is deliberately not stored.** It is exactly
`k * ROLE_REWARD[OWN] + ROLE_REWARD[cause]` -- a pure function of (k,
cause) and the four reward constants. `docs/design-decisions.md` says
those four values "live outside training entirely" and can be changed at
scoring time with no retraining; baking them into a saved column
contradicted that. `reward_for` derives it instead, so changing a reward
constant reprices an existing rollout set for free.

**Perspective is stored resolved.** `generate_training_data.py` samples
some examples from the second team's `OpponentBoardView`. Rather than
store a flag and re-derive, each row stores the role of every word *as the
acting side sees it*, so `board_from_row` returns a plain `Board` that
featurization and re-simulation can consume unchanged. `swapped` is kept
alongside for diagnostics only; nothing reads it to reconstruct state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codenames.board import BOARD_SIZE, Board, Card, Role

__all__ = [
    "CAUSE_NONE",
    "ROLE_CODES",
    "RolloutBatch",
    "RolloutWriter",
    "board_from_row",
    "clue_vocab_fingerprint",
    "load_rollouts",
    "reward_for",
]

# Explicit, ordered role encoding. Pinned here rather than derived from
# Role's definition order so that reordering the enum can never silently
# reinterpret an already-written rollout set.
ROLE_CODES: dict[Role, int] = {Role.OWN: 0, Role.OPPONENT: 1, Role.NEUTRAL: 2, Role.ASSASSIN: 3}
ROLE_BY_CODE: dict[int, Role] = {code: role for role, code in ROLE_CODES.items()}

# `cause is None` means the rollout hit the k cap with no miss (the
# right-censored outcome class). 255 rather than 4 so it can never be
# confused with a real role code if ROLE_CODES ever grows.
CAUSE_NONE = 255

MANIFEST_NAME = "manifest.json"
_COLUMNS = ("board_words", "board_roles", "revealed", "clue", "guesser", "turn_index", "k", "cause", "board_seed", "swapped")


def clue_vocab_fingerprint(clue_words: list[str]) -> str:
    """Cheap identity for the clue vocabulary a rollout set was written
    against. Stored in the manifest and checked on load: clue indices are
    meaningless against a different vocabulary, and a silent mismatch
    would mislabel every row rather than fail."""
    digest = hashlib.sha256()
    digest.update(str(len(clue_words)).encode())
    for word in clue_words:
        digest.update(word.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def reward_for(k: int, cause: Role | None, role_reward: dict[Role, float]) -> float:
    """The reward this rollout earns under `role_reward` -- derived, never
    stored. See this module's docstring."""
    reward = k * role_reward[Role.OWN]
    if cause is not None:
        reward += role_reward[cause]
    return reward


@dataclass(frozen=True)
class RolloutBatch:
    """One shard's columns. Parallel arrays, all length N."""

    board_words: np.ndarray  # (N, BOARD_SIZE) uint16 -- indices into manifest["board_vocab"]
    board_roles: np.ndarray  # (N, BOARD_SIZE) uint8  -- ROLE_CODES, as the acting side sees them
    revealed: np.ndarray  # (N,) uint32 -- bitmask over the BOARD_SIZE slots
    clue: np.ndarray  # (N,) uint32 -- index into the clue vocabulary
    guesser: np.ndarray  # (N,) uint8 -- index into manifest["guesser_names"]
    turn_index: np.ndarray  # (N,) uint8
    k: np.ndarray  # (N,) uint8
    cause: np.ndarray  # (N,) uint8 -- ROLE_CODES or CAUSE_NONE
    board_seed: np.ndarray  # (N,) int64 -- train_scorer.py splits on this, never on row index
    swapped: np.ndarray  # (N,) bool -- diagnostic only

    def __len__(self) -> int:
        return int(self.board_words.shape[0])


def board_from_row(batch: RolloutBatch, i: int, board_vocab: list[str]) -> Board:
    """Reconstruct row `i`'s board state as a plain `Board`, with roles
    already resolved to the acting side's perspective."""
    words = [board_vocab[idx] for idx in batch.board_words[i]]
    cards = tuple(Card(word=w, role=ROLE_BY_CODE[int(c)]) for w, c in zip(words, batch.board_roles[i]))
    mask = int(batch.revealed[i])
    revealed = {w for slot, w in enumerate(words) if mask >> slot & 1}
    return Board(cards=cards, seed=int(batch.board_seed[i]), revealed=revealed)


class RolloutWriter:
    """Accumulates rows in memory and flushes them as a shard, mirroring
    the shard-per-N layout `scripts/pipeline/generate_training_data.py` already
    used, so an interrupted run keeps whatever shards it finished."""

    def __init__(self, output_dir: Path, capacity: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._reset()

    def _reset(self) -> None:
        n = self.capacity
        self._cols = {
            "board_words": np.empty((n, BOARD_SIZE), dtype=np.uint16),
            "board_roles": np.empty((n, BOARD_SIZE), dtype=np.uint8),
            "revealed": np.empty(n, dtype=np.uint32),
            "clue": np.empty(n, dtype=np.uint32),
            "guesser": np.empty(n, dtype=np.uint8),
            "turn_index": np.empty(n, dtype=np.uint8),
            "k": np.empty(n, dtype=np.uint8),
            "cause": np.empty(n, dtype=np.uint8),
            "board_seed": np.empty(n, dtype=np.int64),
            "swapped": np.empty(n, dtype=bool),
        }
        self._filled = 0

    def __len__(self) -> int:
        return self._filled

    def add(
        self,
        *,
        board,
        board_word_index: dict[str, int],
        clue_index: int,
        guesser_index: int,
        turn_index: int,
        k: int,
        cause: Role | None,
        board_seed: int,
        swapped: bool,
    ) -> None:
        i = self._filled
        words = list(board.words)
        self._cols["board_words"][i] = [board_word_index[w] for w in words]
        self._cols["board_roles"][i] = [ROLE_CODES[board.role_of(w)] for w in words]
        mask = 0
        for slot, word in enumerate(words):
            if board.is_revealed(word):
                mask |= 1 << slot
        self._cols["revealed"][i] = mask
        self._cols["clue"][i] = clue_index
        self._cols["guesser"][i] = guesser_index
        self._cols["turn_index"][i] = turn_index
        self._cols["k"][i] = k
        self._cols["cause"][i] = CAUSE_NONE if cause is None else ROLE_CODES[cause]
        self._cols["board_seed"][i] = board_seed
        self._cols["swapped"][i] = swapped
        self._filled += 1

    def flush(self, shard_index: int) -> int:
        """Write the accumulated rows as shard `shard_index`; returns how
        many were written. A no-op on an empty buffer."""
        if self._filled == 0:
            return 0
        for name, array in self._cols.items():
            np.save(self.output_dir / f"{name}_{shard_index:05d}.npy", array[: self._filled])
        written = self._filled
        self._reset()
        return written

    @property
    def is_full(self) -> bool:
        return self._filled >= self.capacity


def write_manifest(
    output_dir: Path,
    *,
    board_vocab: list[str],
    clue_vocab_hash: str,
    n_clue_words: int,
    guesser_names: list[str],
    guesser_pool_config: str,
    seed: int,
    extra: dict | None = None,
) -> None:
    """The rollout set's self-description. `board_vocab` is stored in full
    (400 words, trivial) so decoding never depends on a wordlist file that
    may have changed since -- exactly the drift that step 5's holdout
    change would otherwise introduce."""
    manifest = {
        "board_vocab": board_vocab,
        "clue_vocab_hash": clue_vocab_hash,
        "n_clue_words": n_clue_words,
        "guesser_names": guesser_names,
        "guesser_pool_config": guesser_pool_config,
        "seed": seed,
    }
    if extra:
        manifest.update(extra)
    (Path(output_dir) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest(rollout_dir: Path) -> dict:
    return json.loads((Path(rollout_dir) / MANIFEST_NAME).read_text())


def shard_indices(rollout_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in Path(rollout_dir).glob("board_words_*.npy"))


def load_rollouts(rollout_dir: Path, shard_index: int) -> RolloutBatch:
    d = Path(rollout_dir)
    cols = {name: np.load(d / f"{name}_{shard_index:05d}.npy") for name in _COLUMNS}
    return RolloutBatch(**cols)
