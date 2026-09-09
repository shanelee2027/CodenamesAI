"""Frozen eval suite + eval store (docs/iteration-architecture.md steps
5-6): evaluation against the LLM guesser is real money and should be
paid for exactly once per (spymaster, suite, board).

**The suite.** Fixed board seeds, built from
`codenames.board.load_holdout_wordlist()` (150 of 400 words, none of
which any model has trained on -- see docs/design-decisions.md's
held-out-board-words note, and docs/iteration-architecture.md step 5 for
why 150 rather than 60), against one fixed LLM guesser. The LLM model id
is recorded as part of `EvalSuite.suite_id` -- switching Haiku to Sonnet
to Opus invalidates every cached response and makes old numbers
non-comparable (see docs/iteration-architecture.md's "two roles for
guessers" section), so it must invalidate the suite identity too.
`board_seeds` is deliberately *not* part of `suite_id`: extending a suite
from 100 to 300 boards should reuse every already-recorded game under
the old 100, not treat the suite as a new one -- see `run_eval_suite`.

**The store.** `codenames.llm_store.GameRecordStore` already writes one
row per game; this module only supplies the right key
(`spymaster_id`, `suite_id`, `board_seed`) and the skip-if-already-
recorded logic that key makes possible -- not a new storage mechanism.

**Model identity.** `scripts/train_scorer.py` always writes
`cache/checkpoints/scorer_best.pt`, so a path-keyed identity would
silently serve one model's expensive eval results as another's the
moment that path is overwritten by a second training run.
`spymaster_identity` hashes the checkpoint's *content* instead --
`spymaster_id` for a trained model is `f"{name}:{content_hash}"`, so two
different checkpoints that happen to share a path never collide, and the
same checkpoint bytes always resolve to the same identity regardless of
where the file lives.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from codenames.board import Role, load_holdout_wordlist
from codenames.game import TurnResult, TwoTeamGameResult, TwoTeamTurnResult
from codenames.llm_store import DEFAULT_DB_PATH, GameRecordStore
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import BatchScoringSpymaster
from codenames.two_team_arena import TwoTeamSelfPlayResult, _new_stats, finalize_result, update_stats
from codenames.two_team_gpu_arena import run_two_team_self_play_gpu

DEFAULT_EVAL_SUITE_CONFIG = Path(__file__).parent.parent / "configs" / "eval_suite.json"


@dataclass(frozen=True)
class EvalSuite:
    name: str
    board_seeds: tuple[int, ...]
    guesser_pool_config: Path
    guesser_name: str
    llm_model: str

    @property
    def suite_id(self) -> str:
        """Identity independent of `board_seeds` on purpose -- see this
        module's docstring. Depends on the *content* of
        `guesser_pool_config`, not just its path, so an edit to the
        guesser config (a different noise seed, a different prompt
        parameter) invalidates the suite rather than silently reusing
        stale results under an unchanged path."""
        payload = {
            "name": self.name,
            "llm_model": self.llm_model,
            "guesser_name": self.guesser_name,
            "guesser_pool_config": Path(self.guesser_pool_config).read_text(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_eval_suite(config: Path = DEFAULT_EVAL_SUITE_CONFIG) -> EvalSuite:
    parsed = json.loads(Path(config).read_text())
    return EvalSuite(
        name=parsed["name"],
        board_seeds=tuple(parsed["board_seeds"]),
        guesser_pool_config=Path(parsed["guesser_pool_config"]),
        guesser_name=parsed["guesser_name"],
        llm_model=parsed["llm_model"],
    )


def checkpoint_content_hash(checkpoint_path: Path) -> str:
    """A trained model's identity component: the sha256 of its checkpoint
    file's raw bytes, not its path -- see this module's docstring."""
    return hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()[:16]


def spymaster_identity(name: str, checkpoint_path: Path | None = None) -> str:
    """A spymaster's identity for the eval store. Baselines (no
    checkpoint -- `centroid`, `linear_scorer`, the planned non-deep-
    learning baseline) are identified by their registry name alone: there
    is nothing to hash and no path to collide on. A trained model's
    identity is its name plus its checkpoint's content hash, so retraining
    a checkpoint at the same path never collides with the old results."""
    if checkpoint_path is None:
        return name
    return f"{name}:{checkpoint_content_hash(checkpoint_path)}"


def _game_result_from_row(row) -> TwoTeamGameResult:
    """Reconstructs just enough of a TwoTeamGameResult to feed
    `codenames.two_team_arena.update_stats` -- the same JSON shape
    `GameRecordStore.add_game` wrote it in."""
    turns = [
        TwoTeamTurnResult(
            team=t["team"],
            turn=TurnResult(
                clue=t["clue"],
                number=t["number"],
                guesses=[(word, Role(role)) for word, role in t["guesses"]],
                ended_reason=t["ended_reason"],
            ),
        )
        for t in json.loads(row["turns"])
    ]
    return TwoTeamGameResult(seed=row["seed"], turns=turns, outcome=row["outcome"], winner=row["winner"])


def run_eval_suite(
    spymaster: BatchScoringSpymaster,
    spymaster_id: str,
    suite: EvalSuite,
    sims: SimilarityTensor,
    game_record_db: Path = DEFAULT_DB_PATH,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> TwoTeamSelfPlayResult:
    """Play every board in `suite.board_seeds` not already recorded for
    `(spymaster_id, suite.suite_id)`, then return stats aggregated across
    every recorded game for that pair (old and newly-played alike).

    A rerun for a model already fully evaluated on this suite performs
    zero LLM API calls and re-simulates zero games -- `store.recorded_seeds`
    is checked *before* `run_two_team_self_play_gpu` is ever called, so
    when nothing is missing the simulation path is skipped entirely, not
    just cache-hit inside it. Extending `suite.board_seeds` from N to M
    boards only simulates the M-N seeds not yet in the store."""
    store = GameRecordStore(game_record_db)
    already = store.recorded_seeds(spymaster_id, suite.suite_id)
    missing = [seed for seed in suite.board_seeds if seed not in already]

    if missing:
        run_two_team_self_play_gpu(
            spymaster,
            guesser_pool_config=suite.guesser_pool_config,
            guesser_name=suite.guesser_name,
            seeds=missing,
            sims=sims,
            batch_size=batch_size,
            device=device,
            game_record_db=game_record_db,
            vocabulary=load_holdout_wordlist(),
            spymaster_id=spymaster_id,
            suite_id=suite.suite_id,
        )

    stats = _new_stats()
    for row in store.games_for_suite(spymaster_id, suite.suite_id):
        update_stats(stats, _game_result_from_row(row))
    return finalize_result(stats)
