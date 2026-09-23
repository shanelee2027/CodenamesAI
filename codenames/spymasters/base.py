"""Spymaster interface (docs/iteration-architecture.md
step 1).

A spymaster picks a (clue, number) pair for the current board state.
Every spymaster implements this same interface so the arena can play any
spymaster against any guesser without special-casing.

`TurnContext` bundles the board state with a turn counter instead of
passing loose (board, sims) arguments. Every model is board-state-only
today and ignores `turn_index`; a model that wanted it would otherwise
reconstruct it as `len(board.revealed)`, letting the caller and the model
disagree in principle about what turn it is. Threading it through
explicitly from the game loop removes that risk and gives future models
(clue history, past guesses) somewhere to grow without touching every
existing model or caller.

`top_clues` is the primary method -- score, rank, and pick the best k
legal (clue, number, score) triples -- and `give_clue` is a thin wrapper
around it (`top_clues(ctx, sims, 1)[0]`), so there is exactly one scoring
path per model instead of two (a single-pick path and a top-k path) that
can silently drift apart.

Reward parameters (own/neutral/opponent/assassin) live on the model
itself, not here -- the arena never reads them. Legality
filtering stays in `codenames/clue_search.py`: it's a rule of Codenames,
identical for every model, and must not be reimplemented per model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from codenames.board import MAX_CLUE_NUMBER, Board, OpponentBoardView
from codenames.similarity import SimilarityTensor

__all__ = ["Spymaster", "TurnContext", "MAX_CLUE_NUMBER"]


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

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        """Files the model loads that its constructor params don't name
        exhaustively -- hashed into its eval identity
        (codenames/eval_suite.py::spymaster_identity), so retraining one in
        place makes a different model. None for a model with no file."""
        return []
