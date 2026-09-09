"""Baseline: threshold-free expected-words-covered spymaster.

Replaces `z_threshold.py` (see `docs/versions/z_threshold.md`), whose hard
role thresholds needed a three-stage fallback chain for the "no clue
clears every threshold" case, and whose expected-reward term was
measurably inert on the normal (non-fallback) path -- the thresholds
already rejected every risky candidate, so among survivors the risk term
had nothing left to discriminate on. This model has no thresholds and no
fallback chain: every candidate `(clue, k)` pair gets a finite score, so
`argmax` always has an answer.

**The metric.** For a clue with `a_1 >= a_2 >= ...` the descending
z-scores against unrevealed own words, and `b_w` the z-score against each
unrevealed non-own word `w` (cost `c_w = abs(ROLE_REWARD[role(w)])`,
imported rather than hardcoded so a future reward retune doesn't silently
desync this file):

    q_w(x, tau) = Phi((b_w - x) / tau)          # torch.special.ndtr
    s_j     = prod_w (1 - q_w(a_j, tau_gain))     for j = 1..k
    gain(k) = sum_{j=1..k} prod_{i<=j} s_i
    penalty(k) = sum_w c_w * q_w(a_k, tau_pen)
    score(clue, k) = gain(k) - penalty(k)

`s_j` depends only on `j`, never on the outer `k` -- it is "how likely is
the guesser to still be on an own word by the time they reach the j-th
one," a survival probability. `gain(k)` sums those survival-weighted
increments for j=1..k, which is deliberately *sub-linear* in k: the k-th
word's marginal contribution is `prod_{i<=k} s_i <= 1`, so announcing a
larger number only helps if the words involved are actually distinct
enough (in z-score) from the distractors to keep that product near 1. A
plain `k - penalty` was tried and always picked k=4 (see
`docs/versions/z_threshold.md`'s "expected-reward term is currently
inert" section for the analogous failure in the old model, and this
file's own version doc for the linear variant's fixed measurement) --
this is why that shortcut is refused here.

`penalty(k)` uses only `a_k`, the *weakest* of the intended k words: it
asks, for each non-own word, how likely a guesser who has correctly
reached the k-th intended word would then also drift onto that
distractor. Both terms use the same `diff = b_w - a` tensor, just with
different `tau` (`tau_gain` for the survival term, `tau_pen` for the risk
term) -- kept as one tensor rather than two so the two views of "how close
is this distractor to this own-word z" can't silently drift apart.

**Selection is joint, not per-clue-then-per-k.** For a fixed clue, `s_j`
doesn't depend on k, so per clue the best k is unambiguous (whichever
maximizes `gain(k) - penalty(k)`); reducing each clue to its own best
`(k, score)` before ranking across clues therefore loses nothing relative
to a literal argmax over the full `(clue, k)` grid -- it's the same
maximum, computed without building an explicit product-of-vocabulary
tuple list. What must *not* happen is fixing k by a rule independent of
score (as `z_threshold.py` did, counting how many own words clear a
percentile bar) and only afterward asking "is this a good idea" -- here k
is chosen by the same objective that ranks clues against each other.

**No thresholds, no fallback chain.** `t`, `neutral_outside`,
`opponent_outside`, `assassin_outside`, and `guesser_noise_std` from the
old model don't exist here; every candidate has a finite score, so there
is no "no valid clue" state to fall back out of. The only remaining guard
is the one real degenerate case, a team with zero unrevealed own words
(cannot occur in an actual game, since a team is never asked for a clue
once it has none left) -- handled by returning the most common legal clue
at number=1, matching every other baseline's "never raise, never return
no clue" contract.

Implements `BatchScoringSpymaster` for the same reason `z_threshold.py`
did: `to_device` is a no-op (numpy/CPU only) and `score_batch` loops the
same per-context scoring `top_clues` uses, so `codenames/gpu_arena.py`/
`two_team_gpu_arena.py` can drive this model without a special case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from codenames.board import Board, OpponentBoardView, Role, is_legal_clue
from codenames.clue_search import top_legal_clue
from codenames.clue_stats import ClueStats
from codenames.game import ROLE_REWARD
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

from .base import MAX_CLUE_NUMBER, Spymaster, TurnContext

__all__ = ["ExpectedWordsSpymaster", "gain_and_penalty"]


def gain_and_penalty(
    a: np.ndarray, b: np.ndarray, costs: np.ndarray, tau_gain: float, tau_pen: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(gain, penalty)`, each `(n_cand, K_max)`, from `a` (`(n_cand,
    K_max)`, descending own z-scores `a_1 >= a_2 >= ...` per candidate),
    `b` (`(n_cand, n_non_own)`, non-own z-scores), and `costs`
    (`(n_non_own,)`, `abs(ROLE_REWARD[role(w)])`). Column `m` is
    `k = m + 1` -- see the module docstring for the derivation. Pulled out
    of `_score_all_clues` as a pure function (no `ClueStats`/board
    lookups) so the metric's algebra -- the sub-linear `gain` term
    especially -- can be unit-tested directly against hand-computed
    z-scores, without a synthetic `SimilarityTensor`/`ClueStats` fixture
    in the way."""
    # diff[:, m, w] = b_w - a_{m+1} -- shared by both terms below (see
    # module docstring on why one tensor, two taus).
    diff = b[:, None, :] - a[:, :, None]  # (n_cand, K_max, n_non_own)

    q_gain = torch.special.ndtr(torch.from_numpy((diff / tau_gain).astype(np.float32))).numpy()
    s = np.prod(1.0 - q_gain, axis=2)  # (n_cand, K_max): s[:, m] = s_{m+1}
    cumprod_s = np.cumprod(s, axis=1)
    gain = np.cumsum(cumprod_s, axis=1)  # gain[:, m] = gain(k=m+1)

    q_pen = torch.special.ndtr(torch.from_numpy((diff / tau_pen).astype(np.float32))).numpy()
    penalty = np.sum(q_pen * costs[None, None, :], axis=2)  # (n_cand, K_max): penalty[:, m] = penalty(k=m+1)

    return gain, penalty


class ExpectedWordsSpymaster(Spymaster):
    def __init__(
        self,
        space: str = "numberbatch",
        tau_gain: float = 2.5,
        tau_pen: float = 0.7,
        max_rarity: float = 10.0,
        *,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        clue_stats: ClueStats | None = None,
    ):
        """`cache_dir`/`clue_stats` aren't tunable parameters -- `clue_stats`
        is a dependency-injection hook so tests can hand this a small
        synthetic `ClueStats` (mean=0, std=1, so raw values equal
        z-scores) instead of loading the real ~2.7MB cached artifact.
        Real callers (the registry, the arena) get the default: load once
        from `cache_dir` at construction, since `codenames/arena.py`
        constructs a fresh spymaster inside every spawned worker
        process."""
        self.space = space
        self.tau_gain = tau_gain
        self.tau_pen = tau_pen
        self.max_rarity = max_rarity
        self.clue_stats = clue_stats if clue_stats is not None else ClueStats.load(cache_dir=cache_dir)

    def to_device(self, device) -> None:
        """No-op: this model is numpy/CPU only, see the module
        docstring's `BatchScoringSpymaster` note."""
        return None

    def _score_all_clues(
        self, board: Board | OpponentBoardView, sims: SimilarityTensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(best_n, scores, margin)`, each `(n_clues,)`, indexed by
        `sims.clue_words`. Each entry is already reduced to that clue's
        own best k (see the module docstring on why that loses nothing
        relative to a literal joint argmax). `margin` (`a_k -
        max_w b_w`, unweighted) exists only for the tie-break the spec
        requires and is not part of the public `score_batch` contract."""
        n_clues = len(sims.clue_words)
        best_n_full = np.ones(n_clues, dtype=np.int64)
        scores_full = np.full(n_clues, -np.inf, dtype=np.float32)
        margin_full = np.zeros(n_clues, dtype=np.float32)

        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        if not own:
            # Defensive only -- a real game never asks a team for a clue
            # once it has no own words left. Never raise; fall back to
            # the most common legal clue with number=1, same contract as
            # z_threshold.py's guard.
            scores_full[:] = -self.clue_stats.rarity_percentile
            return best_n_full, scores_full, margin_full

        neutral = board.words_by_role(Role.NEUTRAL, unrevealed_only=True)
        opponent = board.words_by_role(Role.OPPONENT, unrevealed_only=True)
        assassin = board.words_by_role(Role.ASSASSIN, unrevealed_only=True)
        non_own = neutral + opponent + assassin
        non_own_roles = [Role.NEUTRAL] * len(neutral) + [Role.OPPONENT] * len(opponent) + [Role.ASSASSIN] * len(assassin)
        costs = np.array([abs(ROLE_REWARD[r]) for r in non_own_roles], dtype=np.float32)  # (n_non_own,)

        rarity_ok = self.clue_stats.rarity_percentile <= self.max_rarity
        candidate_idx = np.flatnonzero(rarity_ok)
        if candidate_idx.size == 0:
            # Nothing clears the rarity filter -- it's a pool restriction,
            # not a quality judgment, so relax it rather than come back
            # with no clue at all.
            candidate_idx = np.arange(n_clues)

        z_own = self.clue_stats.z_for_board(sims, own, self.space)[candidate_idx]  # (n_cand, n_own)
        z_non_own = self.clue_stats.z_for_board(sims, non_own, self.space)[candidate_idx]  # (n_cand, n_non_own)

        n_cand = len(candidate_idx)
        K_max = min(len(own), MAX_CLUE_NUMBER)
        a = -np.sort(-z_own, axis=1)[:, :K_max]  # (n_cand, K_max): a[:, m] is a_{m+1}, descending
        b = z_non_own  # (n_cand, n_non_own)

        gain, penalty = gain_and_penalty(a, b, costs, self.tau_gain, self.tau_pen)
        score = gain - penalty  # (n_cand, K_max)

        if b.shape[1] > 0:
            max_b = np.max(b, axis=1)  # (n_cand,)
        else:
            max_b = np.full(n_cand, -np.inf, dtype=np.float32)
        margin = a - max_b[:, None]  # (n_cand, K_max)

        # Reduce each clue to its own best k -- see module docstring:
        # since s_j doesn't depend on the outer k, this is exact, not an
        # approximation of the joint argmax.
        best_m = np.argmax(score, axis=1)  # (n_cand,)
        best_score = np.take_along_axis(score, best_m[:, None], axis=1)[:, 0]
        best_margin = np.take_along_axis(margin, best_m[:, None], axis=1)[:, 0]
        best_number = (best_m + 1).astype(np.int64)

        scores_full[candidate_idx] = best_score
        best_n_full[candidate_idx] = best_number
        margin_full[candidate_idx] = best_margin

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
        index ascending -- same tie-break `z_threshold.py::_pick_top_clues`
        used, reused verbatim since it's a selection-mechanics concern,
        not a scoring one."""
        finite_idx = np.flatnonzero(np.isfinite(scores))
        if finite_idx.size == 0:
            # Should be unreachable (every real candidate gets a finite
            # score; only the "no own words" guard above can leave scores
            # entirely unset, and that path already fills scores_full),
            # but never come back with nothing to try.
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
