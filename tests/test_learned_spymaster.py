from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from codenames.board import Board, Card, Role
from codenames.spymasters.base import TurnContext
from codenames.spymasters.learned import LearnedSpymaster
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


def make_ctx(board: Board, turn_index: int | None = None) -> TurnContext:
    return TurnContext(board=board, turn_index=turn_index if turn_index is not None else len(board.revealed))


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


class TestLearnedSpymaster:
    def test_gives_a_legal_clue_from_the_vocabulary(self, sims, checkpoint_path):
        cm = LearnedSpymaster(checkpoint_path)
        board = make_board()
        clue, number = cm.give_clue(make_ctx(board), sims)
        assert clue in sims.clue_words
        from codenames.board import is_legal_clue

        assert is_legal_clue(clue, board.words)

    def test_number_is_within_the_models_full_range(self, sims, checkpoint_path):
        cm = LearnedSpymaster(checkpoint_path)
        board = make_board()
        _, number = cm.give_clue(make_ctx(board), sims)
        assert 1 <= number <= 4  # floored at 1, like every other spymaster here

    def test_top_k_clues_ranked_best_first_and_agrees_with_give_clue(self, sims, checkpoint_path):
        cm = LearnedSpymaster(checkpoint_path)
        board = make_board()
        top2 = cm.top_clues(make_ctx(board), sims, k=2)
        assert len(top2) == 2
        for clue, number, _ in top2:
            assert clue in sims.clue_words
            assert 1 <= number <= 4
        assert top2[0][2] >= top2[1][2]  # scores non-increasing

        clue, number = cm.give_clue(make_ctx(board), sims)
        assert (clue, number) == top2[0][:2]

    def test_turn_index_comes_from_the_context_not_recomputed(self, sims, checkpoint_path, monkeypatch):
        """docs/iteration-architecture.md step 1: turn_index is threaded
        through TurnContext, not reconstructed internally as
        len(board.revealed) -- pass a turn_index that deliberately
        disagrees with the board's own revealed count and confirm the
        model sees the ctx's value, not a recomputed one."""
        cm = LearnedSpymaster(checkpoint_path)
        board = make_board(revealed=["Board9", "Board10"])  # len(revealed) == 2

        seen_turn_index = {}
        import codenames.spymasters.learned as learned_module

        original = learned_module.build_features_batch

        def spy(board_arg, sims_arg, turn_index):
            seen_turn_index["value"] = turn_index
            return original(board_arg, sims_arg, turn_index)

        monkeypatch.setattr(learned_module, "build_features_batch", spy)
        cm.give_clue(make_ctx(board, turn_index=7), sims)
        assert seen_turn_index["value"] == 7

    def test_score_batch_handles_many_contexts_at_once(self, sims, checkpoint_path):
        """The single-board path (top_clues/give_clue) and a multi-context
        call (as codenames/gpu_arena.py and codenames/two_team_gpu_arena.py
        make) both go through score_batch -- confirm the batched form
        agrees with scoring each context one at a time."""
        cm = LearnedSpymaster(checkpoint_path)
        boards = [make_board(), make_board(revealed=["Board9"])]
        contexts = [make_ctx(b) for b in boards]

        batched = cm.score_batch(sims, contexts)
        individually = [cm.score_batch(sims, [ctx])[0] for ctx in contexts]

        assert len(batched) == 2
        for (best_n_a, scores_a), (best_n_b, scores_b) in zip(batched, individually):
            np.testing.assert_array_equal(best_n_a, best_n_b)
            np.testing.assert_allclose(scores_a, scores_b, rtol=1e-5, atol=1e-6)

    def test_different_risk_aversion_can_change_the_chosen_number(self, sims, checkpoint_path):
        cautious = LearnedSpymaster(checkpoint_path, miss_penalty=-10.0)
        lenient = LearnedSpymaster(checkpoint_path, miss_penalty=-0.01)
        board = make_board()
        # Same underlying model/board -- just confirm both run end-to-end
        # with different knobs without erroring; the knob's effect on a
        # specific clue is already covered by test_scorer.py.
        cautious.give_clue(make_ctx(board), sims)
        lenient.give_clue(make_ctx(board), sims)

    def test_own_neutral_opponent_rewards_are_also_runtime_adjustable(self, sims, checkpoint_path):
        # Same idea as the risk-aversion test above, but for the other 3
        # reward knobs added alongside it -- confirm they're accepted and
        # run end-to-end without erroring (the math itself is covered by
        # test_scorer.py's TestExpectedRewardAndBestN).
        cm = LearnedSpymaster(checkpoint_path, own_reward=2.0, neutral_reward=-0.3, opponent_reward=-2.0)
        board = make_board()
        clue, number = cm.give_clue(make_ctx(board), sims)
        assert clue in sims.clue_words
        assert 1 <= number <= 4
