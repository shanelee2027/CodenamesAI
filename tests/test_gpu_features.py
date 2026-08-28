from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from codenames.board import Board, Card, Role
from codenames.features import build_features_batch
from codenames.similarity import SimilarityTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-batched feature construction needs CUDA")

from codenames.gpu_features import build_features_batch_multi  # noqa: E402

BOARD_WORDS = [f"Board{i}" for i in range(25)]
CLUE_WORDS = ["clueone", "cluetwo", "cluethree"]
SPACES = ["a", "b"]


def make_board(words: list[str] | None = None, revealed: list[str] | None = None, seed: int = 1) -> Board:
    words = words or BOARD_WORDS
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(words, roles))
    board = Board(cards=cards, seed=seed)
    for w in revealed or []:
        board.reveal(w)
    return board


def make_sims(tmp_path, tensor: np.ndarray) -> SimilarityTensor:
    np.save(tmp_path / "similarity_tensor.npy", tensor.astype(np.float16))
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestBuildFeaturesBatchMulti:
    def test_matches_numpy_reference_exactly_for_each_board(self, tmp_path):
        rng = np.random.default_rng(0)
        tensor = rng.random((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES))).astype(np.float32)
        # A real missing-vector case too, not just uniform random values.
        tensor[0, 3, 1] = np.nan
        sims = make_sims(tmp_path, tensor)

        board0 = make_board(seed=1, revealed=["Board0", "Board9"])  # own + opponent revealed
        board1 = make_board(seed=2, revealed=["Board17"])  # a neutral revealed instead

        device = torch.device("cuda")
        batched = build_features_batch_multi(sims, [board0, board1], turn_indices=[2, 1], device=device)
        batched_np = batched.cpu().numpy()

        expected0 = build_features_batch(board0, sims, turn_index=2)
        expected1 = build_features_batch(board1, sims, turn_index=1)

        assert batched_np.shape == (2, len(CLUE_WORDS), expected0.shape[1])
        np.testing.assert_allclose(batched_np[0], expected0, atol=1e-5)
        np.testing.assert_allclose(batched_np[1], expected1, atol=1e-5)

    def test_handles_boards_with_different_numbers_of_unrevealed_words(self, tmp_path):
        # Different revealed counts per role -> different unrevealed-word
        # list lengths per board -- exactly the case the padding/masking
        # logic exists for.
        tensor = np.full((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES)), 0.3, dtype=np.float32)
        sims = make_sims(tmp_path, tensor)

        board_few_revealed = make_board(seed=1, revealed=["Board0"])
        board_many_revealed = make_board(seed=2, revealed=["Board0", "Board1", "Board2", "Board3", "Board4", "Board5", "Board6", "Board7"])

        device = torch.device("cuda")
        batched = build_features_batch_multi(
            sims, [board_few_revealed, board_many_revealed], turn_indices=[1, 8], device=device
        ).cpu().numpy()

        expected_few = build_features_batch(board_few_revealed, sims, turn_index=1)
        expected_many = build_features_batch(board_many_revealed, sims, turn_index=8)

        np.testing.assert_allclose(batched[0], expected_few, atol=1e-5)
        np.testing.assert_allclose(batched[1], expected_many, atol=1e-5)

    def test_tensor_on_gpu_is_cached_by_sims_identity(self, tmp_path):
        from codenames.gpu_features import _TENSOR_GPU_CACHE, tensor_on_gpu

        tensor = np.full((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES)), 0.3, dtype=np.float32)
        sims = make_sims(tmp_path, tensor)
        device = torch.device("cuda")

        t1 = tensor_on_gpu(sims, device)
        t2 = tensor_on_gpu(sims, device)
        assert t1 is t2
        del _TENSOR_GPU_CACHE[id(sims)]  # don't leak this fixture's tensor into other tests
