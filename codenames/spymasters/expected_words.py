"""Baseline: threshold-free expected-words-covered spymaster.

Replaces the earlier `z_threshold.py` (its version doc was removed with
it; the measurements survive in `docs/log.md`), whose hard
role thresholds needed a three-stage fallback chain for the "no clue
clears every threshold" case, and whose expected-reward term was
measurably inert on the normal (non-fallback) path -- the thresholds
already rejected every risky candidate, so among survivors the risk term
had nothing left to discriminate on. This model has no thresholds and no
fallback chain: every candidate `(clue, k)` pair gets a finite score, so
`argmax` always has an answer.

**Notation.** For a clue, `a_1 >= a_2 >= ...` are the descending z-scores
against unrevealed own words, and `b_w` the z-score against each
unrevealed non-own word `w`, costing `c_w = abs(ROLE_REWARD[role(w)])` --
imported rather than hardcoded so a future reward retune doesn't silently
desync this file.

**The guesser model.** One assumption, one parameter. The guesser
perceives word `i` as `z_i + eps_i` with `eps ~ N(0, sigma)` drawn
independently per word, then works down its own perceived order until it
picks a non-own word. `sigma` is therefore "how far off the guesser's read
of any single word is," in z units -- a statement about a listener, which
can be judged by watching clues, rather than a free-floating shape
constant.

Everything follows from conditioning on `D`, the perceived score of the
strongest distractor:

    F(d)   = prod_w Phi((d - b_w) / sigma)      # exact CDF of D
    P(all of 1..j survive | D = d) = prod_{i<=j} Phi((a_i - d) / sigma)
    gain(k)    = INT F'(d) * sum_{j<=k} prod_{i<=j} Phi((a_i - d)/sigma) dd
    penalty(k) = INT F'(d) * (1 - prod_{i<=k} Phi((a_i - d)/sigma)) * cbar(d) dd
    score(clue, k) = gain(k) - penalty(k)

Given `d` the intended words are independent, which is the entire reason
for conditioning on it: the obvious formulation -- multiply each word's
survival probability together -- is wrong twice over, since comparisons
against one intended word share that word's `eps`, and all intended words
face the same distractor draws. Both dependencies are positive, so the
naive product *understates* survival, measured at 0.37 expected words too
low at sigma=1.8. See `gain_and_penalty` and the version doc.

`gain(k)` is deliberately *sub-linear* in k: the k-th word's marginal
contribution is `P(all of 1..k survive) <= 1`, so claiming more only helps
when the words involved are far enough clear of the distractors to keep
that probability near 1. A plain `k - penalty` was tried and always picked
k=4 -- see the version doc -- which is why that shortcut is refused here.

`penalty(k)` charges the cost of the word the guesser would *actually*
pick on a miss. Since it works down its own order, that word is by
definition the strongest distractor, so `cbar(d)` is the expected cost
given the max landed at `d`, obtained by splitting the max's mass across
which word achieved it. Summing `c_w` over every distractor that might
have broken through would count several misses that cannot all happen.

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
old model don't exist here. Neither does the earlier `tau_gain`/`tau_pen`
pair: two widths implied two different guessers, and after conditioning on
D the algebra is written in `sigma` throughout, so no `tau` survives to
name. Every candidate has a finite score, so there is no "no valid clue"
state to fall back out of. The only remaining guard
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


GRID_CELLS = 96
GRID_PAD = 6.0


def _ndtr(x: np.ndarray) -> np.ndarray:
    return torch.special.ndtr(torch.from_numpy(x.astype(np.float32))).numpy()


def gain_and_penalty(
    a: np.ndarray, b: np.ndarray, costs: np.ndarray, sigma: float, cells: int = GRID_CELLS
) -> tuple[np.ndarray, np.ndarray]:
    """`(gain, penalty)`, each `(n_cand, K_max)`, from `a` (`(n_cand,
    K_max)`, descending own z-scores per candidate), `b` (`(n_cand,
    n_non_own)`, non-own z-scores) and `costs` (`(n_non_own,)`,
    `abs(ROLE_REWARD[role(w)])`). Column `m` is `k = m + 1`.

    Both terms are exact expectations under the guesser model (the module
    docstring derives it): the guesser perceives word `i` as `z_i + eps_i`
    with `eps ~ N(0, sigma)` drawn independently per word, and works down
    its own perceived order until it picks a non-own word.

    The whole thing turns on conditioning on **D**, the perceived score of
    the strongest distractor. The obvious formulation -- multiply each
    word's survival probability together -- is wrong twice over, because
    those events are not independent: every comparison against one intended
    word shares that word's `eps`, and every intended word faces the same
    distractor draws. Both dependencies are positive, so the naive product
    *understates* survival; measured against Monte Carlo it was low by 0.37
    expected words at sigma=1.8 (see docs/versions/expected_words.md).

    Conditioning on `D = d` removes both at once, because given `d` each
    intended word independently survives with probability
    `Phi((a_i - d) / sigma)`. `D` is a maximum of independents, so its CDF
    is available in closed form, `F(d) = prod_w Phi((d - b_w) / sigma)`,
    and a grid over `d` taking each cell's mass as `F(d_hi) - F(d_lo)`
    integrates it without needing the density. This reproduces Monte Carlo
    to 3-4 decimals.

    Pure function (no `ClueStats`/board lookups) so the algebra can be
    unit-tested directly against hand-computed z-scores.
    """
    n_cand, k_max = a.shape

    # One grid per candidate clue, spanning where that clue's own D can
    # plausibly land. Padding by GRID_PAD sigma on each side puts the
    # unresolved tail mass far below float32's resolution.
    lo = b.min(axis=1) - GRID_PAD * sigma  # (n_cand,)
    hi = np.maximum(b.max(axis=1), a[:, 0]) + GRID_PAD * sigma
    steps = np.linspace(0.0, 1.0, cells + 1, dtype=np.float32)
    edges = lo[:, None] + (hi - lo)[:, None] * steps[None, :]  # (n_cand, cells+1)

    # F[:, e] = P(D <= edges[:, e]) -- exact CDF of the max distractor.
    F = _ndtr((edges[:, :, None] - b[:, None, :]) / sigma).prod(axis=2)  # (n_cand, cells+1)
    mass = np.diff(F, axis=1)  # (n_cand, cells)
    mid = 0.5 * (edges[:, :-1] + edges[:, 1:])  # (n_cand, cells)

    # surv[:, c, m] = P(intended word m+1 outranks D | D = mid_c)
    surv = _ndtr((a[:, None, :] - mid[:, :, None]) / sigma)  # (n_cand, cells, K_max)
    joint = np.cumprod(surv, axis=2)  # all of 1..k survive, given d

    # gain(k) = sum_{j<=k} P(all of 1..j survive), integrated over d.
    per_j = np.einsum("nc,nck->nk", mass, joint)
    gain = np.cumsum(per_j, axis=1)

    # A miss happens exactly when some intended word fails to outrank D,
    # and the word the guesser then picks IS the strongest distractor -- so
    # the cost is that word's, not a sum over every distractor that might
    # have broken through. `resp[:, c, w]` is the chance w is the one at
    # D = mid_c: the max's mass, split by which word achieves it.
    hazard = np.exp(-0.5 * ((mid[:, :, None] - b[:, None, :]) / sigma) ** 2)
    hazard /= np.maximum(_ndtr((mid[:, :, None] - b[:, None, :]) / sigma), 1e-12)
    resp = hazard / np.maximum(hazard.sum(axis=2, keepdims=True), 1e-12)  # (n_cand, cells, n_non_own)
    cost_at_d = resp @ costs  # (n_cand, cells): expected cost of a miss at D = d
    penalty = np.einsum("nc,nck->nk", mass * cost_at_d, 1.0 - joint)

    return gain.astype(np.float32), penalty.astype(np.float32)


class ExpectedWordsSpymaster(Spymaster):
    def __init__(
        self,
        space: str = "numberbatch",
        sigma: float = 1.8,
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
        self.sigma = sigma
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

        gain, penalty = gain_and_penalty(a, b, costs, self.sigma)
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
