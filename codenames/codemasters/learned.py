"""The learned codemaster (SCOPE.md §M8): play-time scoring built on the
trained Scorer, with a runtime risk-aversion parameter.

Scores every candidate clue in one batched forward pass, per SCOPE §2:
`build_features_batch` gathers+sorts the whole clue vocabulary against the
current board in one vectorized pass (no per-clue Python loop), the model
scores all of them in one forward pass, and `expected_reward_and_best_n`
(see codenames/scorer.py) turns that into a (best_n, score) pair per clue
using the current `miss_penalty` -- adjustable per instance, at any time,
with no retraining, since the model itself was never trained against any
particular penalty value.

`turn_index` isn't part of the Codemaster interface (`give_clue(board,
sims)` -- no turn counter is threaded through the arena/game loop). Uses
the same proxy `generate_training_data.py` used to label training examples
(count of currently-revealed words) -- using a different proxy at play time
than at training time would be a silent train/serve skew.
"""

from __future__ import annotations

from pathlib import Path

import torch

from codenames.board import Board
from codenames.clue_search import top_k_legal_clues, top_legal_clue
from codenames.features import build_features_batch
from codenames.scorer import DEFAULT_MISS_PENALTY, Scorer, expected_reward_and_best_n
from codenames.similarity import SimilarityTensor

from .base import Codemaster


class LearnedCodemaster(Codemaster):
    def __init__(self, checkpoint_path: Path | str, miss_penalty: float = DEFAULT_MISS_PENALTY, device: str = "cpu"):
        self.miss_penalty = miss_penalty
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        self.model = Scorer(input_dim=checkpoint["input_dim"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _score_all_clues(self, board: Board, sims: SimilarityTensor) -> tuple:
        """(best_n, score) per clue in sims.clue_words -- shared by
        give_clue() and top_k_clues() so both use the exact same forward
        pass and the exact same current miss_penalty."""
        turn_index = len(board.revealed)
        features = build_features_batch(board, sims, turn_index)

        with torch.no_grad():
            x = torch.from_numpy(features).to(self.device)
            probs = self.model.predict_proba(x).cpu().numpy()

        return expected_reward_and_best_n(probs, self.miss_penalty)

    def give_clue(self, board: Board, sims: SimilarityTensor) -> tuple[str, int]:
        best_n, scores = self._score_all_clues(board, sims)
        clue = top_legal_clue(sims, board, scores)
        clue_idx = sims.clue_index[clue.lower()]
        return clue, int(best_n[clue_idx])

    def top_k_clues(self, board: Board, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        """Up to k best legal (clue, number, score) triples, best first --
        for inspecting what the model likes rather than just its single
        pick (scripts/web_inspector.py)."""
        best_n, scores = self._score_all_clues(board, sims)
        clues = top_k_legal_clues(sims, board, scores, k)
        return [(clue, int(best_n[sims.clue_index[clue.lower()]]), float(scores[sims.clue_index[clue.lower()]])) for clue in clues]
