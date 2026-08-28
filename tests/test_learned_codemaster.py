from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from codenames.board import Board, Card, Role
from codenames.codemasters.learned import LearnedCodemaster
from codenames.features import feature_dim
from codenames.scorer import Scorer
from codenames.similarity import SimilarityTensor

BOARD_WORDS = [f"Board{i}" for i in range(25)]
CLUE_WORDS = ["clueone", "cluetwo", "cluethree"]
SPACES = ["a", "b"]


def make_board(revealed: list[str] | None = None) -> Board:
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(BOARD_WORDS, roles))
    board = Board(cards=cards, seed=1)
    for w in revealed or []:
        board.reveal(w)
    return board


@pytest.fixture
def sims(tmp_path):
    rng = np.random.default_rng(0)
    tensor = rng.random((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES))).astype(np.float16)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


@pytest.fixture
def checkpoint_path(tmp_path):
    dim = feature_dim(len(SPACES))
    model = Scorer(input_dim=dim)
    path = tmp_path / "scorer.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": dim}, path)
    return path


class TestLearnedCodemaster:
    def test_gives_a_legal_clue_from_the_vocabulary(self, sims, checkpoint_path):
        cm = LearnedCodemaster(checkpoint_path)
        board = make_board()
        clue, number = cm.give_clue(board, sims)
        assert clue in sims.clue_words
        from codenames.board import is_legal_clue

        assert is_legal_clue(clue, board.words)

    def test_number_is_within_the_models_full_range(self, sims, checkpoint_path):
        cm = LearnedCodemaster(checkpoint_path)
        board = make_board()
        _, number = cm.give_clue(board, sims)
        assert 0 <= number <= 4  # unlike the baselines, 0 is a legitimate output here

    def test_top_k_clues_ranked_best_first_and_agrees_with_give_clue(self, sims, checkpoint_path):
        cm = LearnedCodemaster(checkpoint_path)
        board = make_board()
        top2 = cm.top_k_clues(board, sims, k=2)
        assert len(top2) == 2
        for clue, number, _ in top2:
            assert clue in sims.clue_words
            assert 0 <= number <= 4
        assert top2[0][2] >= top2[1][2]  # scores non-increasing

        clue, number = cm.give_clue(board, sims)
        assert (clue, number) == top2[0][:2]

    def test_turn_index_proxy_is_revealed_count(self, sims, checkpoint_path, monkeypatch):
        cm = LearnedCodemaster(checkpoint_path)
        board = make_board(revealed=["Board9", "Board10"])

        seen_turn_index = {}
        import codenames.codemasters.learned as learned_module

        original = learned_module.build_features_batch

        def spy(board_arg, sims_arg, turn_index):
            seen_turn_index["value"] = turn_index
            return original(board_arg, sims_arg, turn_index)

        monkeypatch.setattr(learned_module, "build_features_batch", spy)
        cm.give_clue(board, sims)
        assert seen_turn_index["value"] == 2

    def test_different_risk_aversion_can_change_the_chosen_number(self, sims, checkpoint_path):
        cautious = LearnedCodemaster(checkpoint_path, miss_penalty=-10.0)
        lenient = LearnedCodemaster(checkpoint_path, miss_penalty=-0.01)
        board = make_board()
        # Same underlying model/board -- just confirm both run end-to-end
        # with different knobs without erroring; the knob's effect on a
        # specific clue is already covered by test_scorer.py.
        cautious.give_clue(board, sims)
        lenient.give_clue(board, sims)
