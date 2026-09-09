"""Baseline: z-scored threshold spymaster.

For every candidate clue, the cached per-clue mean/std
(`codenames/clue_stats.py`) turns raw cosine similarity into a z-score --
"how unusual is this clue's similarity to this particular board word,
relative to how the clue behaves across the whole board vocabulary."
Fixed percentile budgets (own/neutral/opponent/assassin) convert once, at
construction, into z thresholds via `statistics.NormalDist().inv_cdf`
(stdlib -- scipy is not a declared dependency, see `pyproject.toml`): a
clue is a candidate to give if enough own words clear the "own" threshold
and, ideally, every distractor stays under its own (role-specific, since
an assassin is far more dangerous than a neutral at the same z) threshold.

**Risk instead of a hard cutoff for margin.** Thresholds alone would treat
a distractor at z=1.29 (just under a 0.10-tail cutoff) as perfectly safe
and one at z=1.31 as catastrophic, which is a false cliff -- the guesser's
own noise means neither is actually certain. Step 4 below fixes that by
converting the z-margin between the weakest intended own word and each
distractor back into cosine similarity (multiplying by the clue's own std
-- the space the guesser's noise lives in, since noise is added to raw
similarity, not to a z-score), then asking what fraction of that noise
distribution would be enough to lift the distractor above the weakest
intended word. `torch.special.ndtr` gives that probability without a
Python loop over `math.erf`, since a clue has to be checked against up to
24 distractors and ~11k candidate clues survive the rarity filter by
default. Weighting each distractor's probability by
`abs(codenames.game.ROLE_REWARD[role])` is what makes an assassin at a
given margin worth 10x a neutral at the same margin: the risk term is a
deliberately cheap stand-in for the same expected-reward objective
`codenames/scorer.py::expected_reward_and_best_n` computes properly (that
one has an actual learned P(k, cause | clue) distribution to integrate
over; this one assumes the guesser's noise is the only source of
uncertainty and that noise is Gaussian in cosine-similarity space).

Selection is completely deterministic -- no RNG anywhere, matching every
other baseline's `state_rng`-or-nothing discipline, but here there isn't
even a need for `state_rng`: the score is a pure function of the board
state and the cached statistics. This matters because
`docs/iteration-architecture.md` step 6's eval store replays cached games
and re-bills real LLM calls the moment a rerun's clue diverges from what
was recorded.

Implements `BatchScoringSpymaster` (`codenames/spymasters/base.py`) even
though there is no GPU work to batch -- `to_device` is a no-op and
`score_batch` just loops calling the same per-context scoring this
model's `top_clues` uses, so `codenames/gpu_arena.py`/
`two_team_gpu_arena.py` can drive this model exactly like `LearnedSpymaster`
without a special case.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from codenames.board import Board, OpponentBoardView, Role, is_legal_clue
from codenames.clue_search import top_legal_clue
from codenames.clue_stats import ClueStats
from codenames.game import ROLE_REWARD
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

from .base import MAX_CLUE_NUMBER, Spymaster, TurnContext

__all__ = ["ZThresholdSpymaster"]


class ZThresholdSpymaster(Spymaster):
    def __init__(
        self,
        space: str = "numberbatch",
        own_top: float = 0.10,
        neutral_outside: float = 0.10,
        opponent_outside: float = 0.20,
        assassin_outside: float = 0.30,
        max_rarity: float = 10.0,
        guesser_noise_std: float = 0.03,
        *,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        clue_stats: ClueStats | None = None,
    ):
        """`cache_dir`/`clue_stats` aren't part of the tunable-parameter
        set the spec calls out (those seven above are) -- `clue_stats` is
        a dependency-injection hook so tests can hand this a small
        synthetic `ClueStats` instead of loading the real ~2.7MB cached
        artifact (which itself depends on the real, gitignored 256MB
        similarity tensor). Real callers (the registry, the arena) get
        the default: load once from `cache_dir` at construction, which
        happens once per spymaster instance rather than once per turn --
        important since `codenames/arena.py` constructs a fresh instance
        inside every spawned worker process."""
        self.space = space
        self.own_top = own_top
        self.neutral_outside = neutral_outside
        self.opponent_outside = opponent_outside
        self.assassin_outside = assassin_outside
        self.max_rarity = max_rarity
        self.guesser_noise_std = guesser_noise_std

        # Percentile -> z threshold, once, at construction (not per turn,
        # not per clue) -- stdlib NormalDist, not scipy (undeclared dep).
        dist = NormalDist()
        self.t_thresh = dist.inv_cdf(1 - own_top)
        self.n_thresh = dist.inv_cdf(1 - neutral_outside)
        self.o_thresh = dist.inv_cdf(1 - opponent_outside)
        self.a_thresh = dist.inv_cdf(1 - assassin_outside)

        self.clue_stats = clue_stats if clue_stats is not None else ClueStats.load(cache_dir=cache_dir)

    def to_device(self, device) -> None:
        """No-op: this model is numpy/CPU only, see the module
        docstring's `BatchScoringSpymaster` note."""
        return None

    def _score_all_clues(
        self, board: Board | OpponentBoardView, sims: SimilarityTensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(best_n, scores, margin)`, each `(n_clues,)`, indexed by
        `sims.clue_words`. `margin` exists only to break ties the way the
        spec requires (`weakest - max(non-own z)`, larger wins) -- it is
        not part of the public `score_batch` contract, so callers that
        only care about that protocol (`score_batch` below) drop it."""
        n_clues = len(sims.clue_words)
        best_n_full = np.ones(n_clues, dtype=np.int64)
        scores_full = np.full(n_clues, -np.inf, dtype=np.float32)
        margin_full = np.full(n_clues, -np.inf, dtype=np.float32)

        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        neutral = board.words_by_role(Role.NEUTRAL, unrevealed_only=True)
        opponent = board.words_by_role(Role.OPPONENT, unrevealed_only=True)
        assassin = board.words_by_role(Role.ASSASSIN, unrevealed_only=True)

        if not own:
            # Defensive only -- a real game never asks a team for a clue
            # once it has no own words left to give one for. Never raise;
            # fall back to the most common legal clue with number=1.
            scores_full[:] = -self.clue_stats.rarity_percentile
            return best_n_full, scores_full, margin_full

        rarity_ok = self.clue_stats.rarity_percentile <= self.max_rarity
        candidate_idx = np.flatnonzero(rarity_ok)
        if candidate_idx.size == 0:
            # Nothing clears the rarity filter -- relax it rather than
            # come back with no clue at all.
            candidate_idx = np.arange(n_clues)

        non_own = neutral + opponent + assassin
        non_own_roles = [Role.NEUTRAL] * len(neutral) + [Role.OPPONENT] * len(opponent) + [Role.ASSASSIN] * len(assassin)
        non_own_thresholds = np.array(
            [self.n_thresh] * len(neutral) + [self.o_thresh] * len(opponent) + [self.a_thresh] * len(assassin),
            dtype=np.float32,
        )
        non_own_abs_reward = np.array([abs(ROLE_REWARD[r]) for r in non_own_roles], dtype=np.float32)

        si = self.clue_stats.space_index(self.space)
        sd_cand = self.clue_stats.std[candidate_idx, si]  # (n_cand,)

        z_own = self.clue_stats.z_for_board(sims, own, self.space)[candidate_idx]  # (n_cand, n_own)
        z_non_own = self.clue_stats.z_for_board(sims, non_own, self.space)[candidate_idx]  # (n_cand, n_non_own)

        sorted_own = -np.sort(-z_own, axis=1)  # descending, per candidate
        K = np.sum(z_own > self.t_thresh, axis=1)
        k = np.minimum(K, MAX_CLUE_NUMBER)
        eligible = k >= 1
        k_idx = np.clip(k - 1, 0, max(sorted_own.shape[1] - 1, 0))
        weakest = np.take_along_axis(sorted_own, k_idx[:, None], axis=1)[:, 0]

        if z_non_own.shape[1] > 0:
            valid = np.all(z_non_own < non_own_thresholds[None, :], axis=1)
            max_non_own = np.max(z_non_own, axis=1)
        else:
            valid = np.ones(len(candidate_idx), dtype=bool)
            max_non_own = np.full(len(candidate_idx), -np.inf, dtype=np.float32)

        def risk_and_margin(weakest_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            margin = weakest_vec - max_non_own
            if z_non_own.shape[1] == 0:
                risk = np.zeros(len(candidate_idx), dtype=np.float32)
                return risk, margin.astype(np.float32)
            # z-margin * this clue's own std converts back to cosine
            # similarity -- the space the guesser's noise actually lives
            # in -- so sigmas is "how many noise-std's of a boost would
            # this distractor need to overtake the weakest intended word."
            margin_matrix = weakest_vec[:, None] - z_non_own  # (n_cand, n_non_own)
            sigmas = margin_matrix * sd_cand[:, None] / (self.guesser_noise_std * np.sqrt(2.0))
            p = torch.special.ndtr(torch.from_numpy((-sigmas).astype(np.float32))).numpy()
            risk = (p * non_own_abs_reward[None, :]).sum(axis=1)
            return risk.astype(np.float32), margin.astype(np.float32)

        risk, margin = risk_and_margin(weakest)
        score = np.where(eligible, k.astype(np.float32) - risk, -np.inf)

        valid_elig = eligible & valid
        if valid_elig.any():
            chosen_score = np.where(valid_elig, score, -np.inf)
            chosen_k = k
            chosen_margin = margin
        elif eligible.any():
            # Fallback 1: no clue satisfies every role threshold. Relax in
            # order of what a miss actually costs (ROLE_REWARD: neutral
            # -0.2, opponent -1.0, assassin -10.0) rather than dropping
            # every bound at once -- give up the cheap constraints first
            # and only ever the assassin's last. Measured over 1235 turns
            # of real play this branch never executes at the shipped
            # defaults (median 142 valid clues per turn, minimum 1), so
            # this is a safety net, not a hot path; it is graded rather
            # than all-or-nothing so that when it does fire it degrades
            # into a merely-worse clue instead of an instantly-losing one.
            relax_order = (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)
            role_arr = np.array([r.value for r in non_own_roles])
            keep = np.ones(len(non_own_roles), dtype=bool)
            relaxed = eligible
            for role in relax_order:
                keep &= role_arr != role.value
                if keep.any():
                    still_ok = np.all(
                        z_non_own[:, keep] < non_own_thresholds[None, keep], axis=1
                    )
                else:
                    still_ok = np.ones(len(candidate_idx), dtype=bool)
                relaxed = eligible & still_ok
                if relaxed.any():
                    break
            if not relaxed.any():
                relaxed = eligible
            chosen_score = np.where(relaxed, score, -np.inf)
            chosen_k = k
            chosen_margin = margin
        else:
            # Fallback 2: no candidate has any own word above t -- force
            # the intended set to each candidate's single best own word
            # (k=1) instead of giving up.
            weakest_forced = sorted_own[:, 0]
            risk_forced, margin_forced = risk_and_margin(weakest_forced)
            chosen_score = 1.0 - risk_forced
            chosen_k = np.ones(len(candidate_idx), dtype=np.int64)
            chosen_margin = margin_forced

        scores_full[candidate_idx] = chosen_score
        best_n_full[candidate_idx] = chosen_k
        margin_full[candidate_idx] = chosen_margin

        if not np.isfinite(scores_full).any():
            # Unreachable given the fallbacks above, but "never raise,
            # never return no clue" is a hard requirement -- see the
            # module docstring.
            scores_full[:] = -self.clue_stats.rarity_percentile
            best_n_full[:] = 1

        return best_n_full, scores_full, margin_full

    def score_batch(self, sims: SimilarityTensor, contexts: list[TurnContext]) -> list[tuple[np.ndarray, np.ndarray]]:
        """`(best_n, scores)` per context, indexed by `sims.clue_words` --
        the `BatchScoringSpymaster` contract. No GPU batching happens
        (see the module docstring); this loops the same per-context
        scoring `top_clues` uses so there is exactly one implementation."""
        return [self._score_all_clues(ctx.board, sims)[:2] for ctx in contexts]

    def _pick_top_clues(
        self,
        sims: SimilarityTensor,
        board: Board | OpponentBoardView,
        best_n: np.ndarray,
        scores: np.ndarray,
        margin: np.ndarray,
        k: int,
    ) -> list[tuple[str, int, float]]:
        """Up to k legal (clue, number, score) triples, ranked by score
        descending, ties broken by margin descending then clue-vocabulary
        index ascending -- the exact tie-break the spec requires, which
        `codenames.clue_search`'s plain-argsort helpers don't offer, so
        this builds the order itself and defers only legality checking
        (`is_legal_clue`) to that module, per its own contract."""
        finite_idx = np.flatnonzero(np.isfinite(scores))
        if finite_idx.size == 0:
            # Should be unreachable (see _score_all_clues), but never
            # come back with nothing to try.
            finite_idx = np.arange(len(scores))
            scores = -self.clue_stats.rarity_percentile

        order = finite_idx[np.lexsort((finite_idx, -margin[finite_idx], -scores[finite_idx]))]

        picks: list[tuple[str, int, float]] = []
        for idx in order:
            clue = sims.clue_words[idx]
            if is_legal_clue(clue, board.words):
                picks.append((clue, int(best_n[idx]), float(scores[idx])))
                if len(picks) >= k:
                    break

        if not picks:
            # Legality failures are rare (a candidate has to literally
            # contain or be contained by a board word -- see
            # codenames/clue_search.py) but this must never come back
            # empty: fall back to the most common legal clue overall.
            clue = top_legal_clue(sims, board, -self.clue_stats.rarity_percentile)
            idx = sims.clue_index[clue.lower()]
            picks = [(clue, 1, float(scores[idx]) if np.isfinite(scores[idx]) else 0.0)]
        return picks

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        board = ctx.board
        best_n, scores, margin = self._score_all_clues(board, sims)
        return self._pick_top_clues(sims, board, best_n, scores, margin, k)
