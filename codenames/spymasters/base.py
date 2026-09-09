"""Spymaster interface (SCOPE.md §M6/§M8, docs/iteration-architecture.md
step 1).

A spymaster picks a (clue, number) pair for the current board state. The
learned scorer (M8) and every baseline (§6) implement this same interface so
the arena can play any spymaster against any guesser without special-casing.

`TurnContext` bundles the board state with a turn counter instead of
passing loose (board, sims) arguments. Today every model is board-state-
only and ignores `turn_index` except `LearnedSpymaster` (which used to
reconstruct it internally as `len(board.revealed)` -- a train/serve skew
risk its own docstring used to flag, since the caller and the model could
in principle disagree on what turn it is). Threading it through explicitly
from the game loop removes that risk and gives future models (clue
history, past guesses) somewhere to grow without touching every existing
model or caller.

`top_clues` is the primary method -- score, rank, and pick the best k
legal (clue, number, score) triples -- and `give_clue` is a thin wrapper
around it (`top_clues(ctx, sims, 1)[0]`), so there is exactly one scoring
path per model instead of two (a single-pick path and a top-k path) that
can silently drift apart.

Reward parameters (own/neutral/opponent/assassin) live on the model itself
(see `LearnedSpymaster`), not here -- the arena never reads them. Legality
filtering stays in `codenames/clue_search.py`: it's a rule of Codenames,
identical for every model, and must not be reimplemented per model.

`BatchScoringSpymaster` (docs/iteration-architecture.md step 3) is the
protocol a model opts into if it scores the whole clue vocabulary and can
usefully batch that across many simultaneous boards -- currently only
`LearnedSpymaster`. `codenames/gpu_arena.py` and
`codenames/two_team_gpu_arena.py` are written against this protocol only:
they gather board views into `TurnContext`s, call `score_batch` for the
per-clue (best_n, scores) arrays, hand those to `codenames.clue_search`
for the best *legal* clue, and call `to_device` instead of reaching into
model internals directly. A model that doesn't implement this (every
baseline -- they each score a handful of candidates, not the whole
vocabulary, so there's nothing to gain from batching) simply isn't usable
with those two GPU-batched arenas; `codenames/arena.py`'s regular
per-process path works for any `Spymaster`, this one included.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from codenames.board import MAX_CLUE_NUMBER, Board, OpponentBoardView
from codenames.similarity import SimilarityTensor

__all__ = ["Spymaster", "TurnContext", "BatchScoringSpymaster", "MAX_CLUE_NUMBER"]


@dataclass(frozen=True)
class TurnContext:
    board: Board | OpponentBoardView
    turn_index: int


class Spymaster(ABC):
    @abstractmethod
    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        """Up to k best legal (clue, number, score) triples for the
        current board state, best first. `number` is the count of
        own-words the spymaster intends that clue to cover. May return
        fewer than k if the vocabulary doesn't have that many legally-
        scored candidates."""
        raise NotImplementedError

    def give_clue(self, ctx: TurnContext, sims: SimilarityTensor) -> tuple[str, int]:
        """(clue, number) for the current (possibly partially revealed)
        board state -- this model's single best pick."""
        clue, number, _ = self.top_clues(ctx, sims, 1)[0]
        return clue, number


@runtime_checkable
class BatchScoringSpymaster(Protocol):
    """A model that scores the entire clue vocabulary and can batch that
    scoring across many simultaneous boards -- see the module docstring.
    `score_batch`'s per-context arrays are indexed by `sims.clue_words`,
    exactly like the scores `Spymaster.top_clues` hands to
    `codenames.clue_search`; picking the best *legal* clue from them stays
    the arena's job (`codenames.clue_search.top_legal_clue`), not this
    model's."""

    def score_batch(self, sims: SimilarityTensor, contexts: list[TurnContext]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(best_n, scores) per context, indexed by sims.clue_words."""
        ...

    def to_device(self, device) -> None:
        """Move the model (and every future score_batch call) onto
        `device`."""
        ...
