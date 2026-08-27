from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.similarity import SimilarityTensor

CLUE_WORDS = ["apple", "banana", "car"]
BOARD_WORDS = ["Fruit", "Vehicle"]
SPACES = ["space_a", "space_b"]


def write_fake_cache(cache_dir) -> np.ndarray:
    # (n_clues=3, n_board=2, n_spaces=2), values chosen to be distinguishable
    # per-clue/board/space so indexing bugs would show up as test failures.
    tensor = np.array(
        [
            [[0.9, 0.1], [0.2, 0.3]],  # apple
            [[0.8, 0.2], [0.1, 0.4]],  # banana
            [[0.05, 0.05], [0.95, 0.6]],  # car
        ],
        dtype=np.float16,
    )
    np.save(cache_dir / "similarity_tensor.npy", tensor)
    (cache_dir / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (cache_dir / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (cache_dir / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return tensor


@pytest.fixture
def sims(tmp_path):
    write_fake_cache(tmp_path)
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestLoad:
    def test_shapes_match_vocab(self, sims):
        assert sims.tensor.shape == (3, 2, 2)
        assert sims.clue_words == CLUE_WORDS
        assert sims.board_words == BOARD_WORDS
        assert sims.spaces == SPACES

    def test_mismatched_shape_raises(self, tmp_path):
        write_fake_cache(tmp_path)
        (tmp_path / "board_vocab.json").write_text(json.dumps(["Fruit", "Vehicle", "Extra"]))
        with pytest.raises(ValueError):
            SimilarityTensor.load(cache_dir=tmp_path)


class TestSimilarity:
    def test_single_space(self, sims):
        assert sims.similarity("banana", "Fruit", space="space_a") == pytest.approx(0.8, abs=1e-3)

    def test_all_spaces(self, sims):
        result = sims.similarity("car", "Vehicle")
        assert result.shape == (2,)
        assert result[0] == pytest.approx(0.95, abs=1e-3)
        assert result[1] == pytest.approx(0.6, abs=1e-3)

    def test_board_word_is_case_insensitive(self, sims):
        assert sims.similarity("banana", "FRUIT", space="space_a") == pytest.approx(0.8, abs=1e-3)
        assert sims.similarity("banana", "fruit", space="space_a") == pytest.approx(0.8, abs=1e-3)

    def test_clue_is_case_insensitive(self, sims):
        assert sims.similarity("APPLE", "Fruit", space="space_a") == pytest.approx(0.9, abs=1e-3)

    def test_unknown_clue_raises(self, sims):
        with pytest.raises(KeyError):
            sims.similarity("durian", "Fruit")

    def test_unknown_board_word_raises(self, sims):
        with pytest.raises(KeyError):
            sims.similarity("apple", "Not A Board Word")


class TestSimilaritiesForBoard:
    def test_shape_and_values(self, sims):
        result = sims.similarities_for_board("apple", ["Fruit", "Vehicle"], space="space_a")
        assert result.shape == (2,)
        assert result[0] == pytest.approx(0.9, abs=1e-3)
        assert result[1] == pytest.approx(0.2, abs=1e-3)

    def test_all_spaces_shape(self, sims):
        result = sims.similarities_for_board("apple", ["Fruit", "Vehicle"])
        assert result.shape == (2, 2)

    def test_board_words_are_case_insensitive(self, sims):
        result = sims.similarities_for_board("apple", ["FRUIT", "vehicle"], space="space_a")
        assert result[0] == pytest.approx(0.9, abs=1e-3)
        assert result[1] == pytest.approx(0.2, abs=1e-3)

    def test_unknown_board_word_in_list_raises(self, sims):
        with pytest.raises(KeyError):
            sims.similarities_for_board("apple", ["Fruit", "Nope"])


class TestTopClues:
    def test_returns_k_sorted_descending(self, sims):
        top = sims.top_clues("Fruit", k=2, space="space_a")
        assert len(top) == 2
        assert top[0][0] == "apple"  # 0.9 > banana's 0.8 > car's 0.05
        assert top[0][1] > top[1][1]

    def test_default_space_is_first(self, sims):
        top = sims.top_clues("Vehicle", k=1)
        assert top[0][0] == "car"  # space_a value 0.95 is the max in that column
