"""The learned spymaster: play-time scoring built on the
trained Scorer, with four runtime reward parameters.

Scores every candidate clue in one batched forward pass:
feature construction gathers+sorts the whole clue vocabulary against the
current board in one vectorized pass (no per-clue Python loop), the model
scores all of them in one forward pass, and `expected_reward_and_best_n`
(see codenames/scorer.py) turns that into a (best_n, score) pair per clue
using the current `own_reward`/`neutral_reward`/`opponent_reward`/
`miss_penalty` -- each adjustable per instance, at any time, with no
retraining, since the model itself was never trained against any
particular reward value, only against the empirical (k, cause) outcome.
`miss_penalty` keeps that name (rather than `assassin_reward`) since it's
the one meant to double as the risk-aversion knob and the existing
web UI field already calls it that.

`score_batch` (docs/iteration-architecture.md step 3) is the one place
this class does feature construction + forward pass + expected-reward --
`top_clues`/`give_clue` call it with a one-element context list, and
codenames/gpu_arena.py and codenames/two_team_gpu_arena.py call it with
many contexts at once (many simultaneous boards' turns) instead of
reaching into `.model`/`.own_reward`/`.miss_penalty` and calling
`expected_reward_and_best_n` themselves -- one scoring implementation
instead of two that can drift apart.

Feature construction inside `score_batch` branches on device: on CUDA it
uses `codenames.gpu_features.build_features_batch_multi`, which
materializes the whole similarity tensor on-device once per process (fast
-- see that module's docstring for the measured speedup) -- fine for the
GPU arena's one dedicated process per run. On CPU it instead loops
`codenames.features.build_features_batch` per context, which only ever
reads the handful of tensor columns a given board actually needs off the
mmap. That matters concretely for scripts/pipeline/run_arena.py's --no-gpu-batch
fallback, which constructs a LearnedSpymaster fresh inside each of N
spawned CPU worker processes (codenames/arena.py) -- materializing a
private full-tensor copy in every one of them is exactly the RSS blowup
docs/design-decisions.md's memory design note (and
codenames/spymasters/linear_scorer.py's docstring) warns about, so the CPU
path deliberately keeps the mmap-friendly per-board reads instead of
routing through the GPU-oriented batched gather.

`turn_index` comes from the caller's `TurnContext` (see
codenames/spymasters/base.py) -- previously this class reconstructed it
internally as `len(board.revealed)`, the same proxy
`generate_training_data.py` uses to label training examples, but recomputing
it here risked a silent train/serve skew if the two ever diverged. Now the
game loop (codenames/game.py::play_turn) computes it once, the same way,
and threads it through explicitly.
"""

from __future__ import annotations

import numpy as np
import torch

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_search import top_k_legal_clues, top_legal_clue
from codenames.features import build_features_batch
from codenames.game import ROLE_REWARD
from codenames.gpu_features import build_features_batch_multi
from codenames.scorer import DEFAULT_MISS_PENALTY, OWN_REWARD, Scorer, expected_reward_and_best_n
from codenames.similarity import SimilarityTensor

from pathlib import Path

from .base import Spymaster, TurnContext

# Over-fetch pool when a rarity filter is active: the forward pass scoring
# the whole vocabulary already happened, so asking top_k_legal_clues to
# walk a few hundred more already-sorted candidates (rather than just 1)
# is close to free -- this only affects how many get checked for legality
# + the rarity threshold before picking the first survivor.
_RARITY_FETCH_POOL = 300


class LearnedSpymaster(Spymaster):
    def __init__(
        self,
        checkpoint_path: Path | str,
        miss_penalty: float = DEFAULT_MISS_PENALTY,
        own_reward: float = OWN_REWARD,
        neutral_reward: float = ROLE_REWARD[Role.NEUTRAL],
        opponent_reward: float = ROLE_REWARD[Role.OPPONENT],
        device: str = "cpu",
        rarity_percentile: dict[str, float] | None = None,
        max_rarity: float = 100.0,
    ):
        """`rarity_percentile`/`max_rarity` (see codenames/clue_search.py's
        clue_rarity_percentile): default `max_rarity=100.0` means no
        filtering at all, so every existing caller (arena/training scripts)
        is unaffected unless it explicitly opts in -- this is a UI-facing
        knob (scripts/tools/web_inspector.py sets a non-default max_rarity at
        construction), not a change to evaluation methodology."""
        self.miss_penalty = miss_penalty
        self.own_reward = own_reward
        self.neutral_reward = neutral_reward
        self.opponent_reward = opponent_reward
        self.device = torch.device(device)
        self.rarity_percentile = rarity_percentile
        self.max_rarity = max_rarity
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        self.model = Scorer(input_dim=checkpoint["input_dim"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def to_device(self, device: torch.device | str) -> None:
        """Moves the model (and every future score_batch call) onto
        `device` -- replaces an arena reaching in to set
        `spymaster.model.to(device)`/`spymaster.device` directly (see
        codenames/gpu_arena.py, codenames/two_team_gpu_arena.py)."""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self.device)

    def _pick_legal_clue(self, sims: SimilarityTensor, board: Board | OpponentBoardView, scores) -> str:
        """top_legal_clue, unless a rarity filter is active and there's
        actually a percentile table to filter against -- then fetch a
        larger pool of legal candidates and take the best one that clears
        the rarity threshold, falling back to the plain unfiltered pick if
        every candidate in the pool is too rare (never let a UI-only
        filter setting make a real game unable to produce a clue)."""
        if self.max_rarity >= 100.0 or not self.rarity_percentile:
            return top_legal_clue(sims, board, scores)
        candidates = top_k_legal_clues(sims, board, scores, k=_RARITY_FETCH_POOL)
        for clue in candidates:
            if self.rarity_percentile.get(clue, 100.0) <= self.max_rarity:
                return clue
        return top_legal_clue(sims, board, scores)

    def score_batch(self, sims: SimilarityTensor, contexts: list[TurnContext]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(best_n, scores) per context, indexed by sims.clue_words --
        the only place this class builds features, runs the model, and
        turns probabilities into expected reward. See the module
        docstring for why feature construction branches on device."""
        if self.device.type == "cuda":
            boards = [ctx.board for ctx in contexts]
            turn_indices = [ctx.turn_index for ctx in contexts]
            features = build_features_batch_multi(sims, boards, turn_indices, self.device)  # (n, n_clues, dim)
        else:
            stacked = np.stack([build_features_batch(ctx.board, sims, ctx.turn_index) for ctx in contexts])
            features = torch.from_numpy(stacked).to(self.device)

        n, n_clues, dim = features.shape
        with torch.no_grad():
            probs = self.model.predict_proba(features.reshape(n * n_clues, dim)).cpu().numpy()
        probs = probs.reshape(n, n_clues, -1)

        return [
            expected_reward_and_best_n(
                probs[i],
                own_reward=self.own_reward,
                neutral_reward=self.neutral_reward,
                opponent_reward=self.opponent_reward,
                assassin_reward=self.miss_penalty,
            )
            for i in range(n)
        ]

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        """Up to k best legal (clue, number, score) triples, best first --
        for inspecting what the model likes rather than just its single
        pick (scripts/tools/web_inspector.py). k==1 (give_clue's own case) goes
        through `_pick_legal_clue` so a rarity filter, if active, applies
        the same way it always did for a single pick; k>1 does not apply
        the rarity filter (unchanged from before this method existed --
        scripts/tools/web_inspector.py applies its own post-hoc filter over a
        larger fetched pool in that case)."""
        board = ctx.board
        best_n, scores = self.score_batch(sims, [ctx])[0]
        if k == 1:
            clue = self._pick_legal_clue(sims, board, scores)
            idx = sims.clue_index[clue.lower()]
            return [(clue, int(best_n[idx]), float(scores[idx]))]
        clues = top_k_legal_clues(sims, board, scores, k)
        return [(clue, int(best_n[sims.clue_index[clue.lower()]]), float(scores[sims.clue_index[clue.lower()]])) for clue in clues]
