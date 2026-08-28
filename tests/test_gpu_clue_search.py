from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from codenames.clue_search import mean_from_columns
from codenames.similarity import SimilarityTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-batched clue search needs CUDA")

from codenames.gpu_clue_search import batched_mean_similarity  # noqa: E402

BOARD_WORDS = [f"Board{i}" for i in range(25)]
CLUE_WORDS = ["clueone", "cluetwo", "cluethree", "cluefour"]
SPACES = ["a", "b"]


def make_sims(tmp_path, tensor: np.ndarray) -> SimilarityTensor:
    np.save(tmp_path / "similarity_tensor.npy", tensor.astype(np.float16))
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestBatchedMeanSimilarity:
    def test_matches_numpy_reference_per_sample(self, tmp_path):
        rng = np.random.default_rng(0)
        tensor = rng.random((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES))).astype(np.float32)
        tensor[0, 3, 1] = np.nan  # a real missing-vector case
        sims = make_sims(tmp_path, tensor)

        word_lists = [["Board0", "Board3"], ["Board1"], ["Board5", "Board6", "Board7"]]
        device = torch.device("cuda")
        batched = batched_mean_similarity(sims, word_lists, device, chunk_size=2)

        for i, words in enumerate(word_lists):
            expected = mean_from_columns([np.asarray(sims.tensor[:, sims.board_index[w.lower()], :], dtype=np.float32) for w in words])
            np.testing.assert_allclose(batched[i], expected, atol=1e-4, equal_nan=True)

    def test_chunking_does_not_change_the_result(self, tmp_path):
        rng = np.random.default_rng(1)
        tensor = rng.random((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES))).astype(np.float32)
        sims = make_sims(tmp_path, tensor)
        word_lists = [[BOARD_WORDS[i % 25]] for i in range(10)]
        device = torch.device("cuda")

        whole = batched_mean_similarity(sims, word_lists, device, chunk_size=100)
        chunked = batched_mean_similarity(sims, word_lists, device, chunk_size=3)
        np.testing.assert_allclose(whole, chunked, atol=1e-5)

    def test_empty_input(self, tmp_path):
        tensor = np.zeros((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES)), dtype=np.float32)
        sims = make_sims(tmp_path, tensor)
        out = batched_mean_similarity(sims, [], torch.device("cuda"))
        assert out.shape == (0, len(CLUE_WORDS))
