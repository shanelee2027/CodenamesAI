from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.board import BOARD_SIZE, ROLE_COUNTS, Board, Card, Role
from codenames.features import (
    ROLE_ORDER,
    SENTINEL,
    FeatureLayout,
    build_features,
    build_features_batch,
    feature_dim,
)
from codenames.similarity import SimilarityTensor

BOARD_WORDS = [f"Board{i}" for i in range(25)]
CLUE_WORDS = ["clueone", "cluetwo"]
SPACES = ["a", "b"]


def make_board(words: list[str] | None = None, revealed: list[str] | None = None) -> Board:
    words = words or BOARD_WORDS
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(words, roles))
    board = Board(cards=cards, seed=1)
    for w in revealed or []:
        board.reveal(w)
    return board


def make_sims(tmp_path, tensor: np.ndarray) -> SimilarityTensor:
    np.save(tmp_path / "similarity_tensor.npy", tensor.astype(np.float16))
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


def base_tensor(value: float = 0.5) -> np.ndarray:
    return np.full((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES)), value, dtype=np.float32)


class TestFeatureDimAndLayout:
    def test_feature_dim_matches_formula(self):
        assert feature_dim(3) == 25 * 3 + 25 + 3
        assert feature_dim(4) == 25 * 4 + 25 + 3

    def test_layout_slices_are_contiguous_and_cover_everything(self):
        layout = FeatureLayout(spaces=["a", "b"])
        assert layout.space_slice("a") == slice(0, 25)
        assert layout.space_slice("b") == slice(25, 50)
        assert layout.mask_slice() == slice(50, 75)
        assert layout.scalar_slice() == slice(75, 78)
        assert layout.size == 78


class TestBuildFeaturesShape:
    def test_output_length_matches_feature_dim(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        vec = build_features(board, "clueone", sims, turn_index=0)
        assert vec.shape == (feature_dim(len(SPACES)),)
        assert vec.dtype == np.float32


class TestPermutationInvariance:
    def test_same_words_and_roles_in_different_card_order_give_identical_features(self, tmp_path):
        tensor = base_tensor()
        rng = np.random.default_rng(0)
        tensor[:] = rng.random(tensor.shape)
        sims = make_sims(tmp_path, tensor)

        roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
        pairs = list(zip(BOARD_WORDS, roles))

        board_a = Board(cards=tuple(Card(w, r) for w, r in pairs), seed=1)
        shuffled = pairs[::-1]  # same (word, role) pairs, reversed order
        board_b = Board(cards=tuple(Card(w, r) for w, r in shuffled), seed=1)

        vec_a = build_features(board_a, "clueone", sims, turn_index=2)
        vec_b = build_features(board_b, "clueone", sims, turn_index=2)
        np.testing.assert_array_equal(vec_a, vec_b)

    def test_revealing_in_different_order_gives_identical_features(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board_a = make_board(revealed=["Board0", "Board1"])
        board_b = make_board(revealed=["Board1", "Board0"])
        vec_a = build_features(board_a, "clueone", sims, turn_index=0)
        vec_b = build_features(board_b, "clueone", sims, turn_index=0)
        np.testing.assert_array_equal(vec_a, vec_b)


class TestMasking:
    def test_mask_is_all_ones_when_nothing_revealed(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        vec = build_features(board, "clueone", sims, turn_index=0)
        layout = FeatureLayout(spaces=SPACES)
        mask = vec[layout.mask_slice()]
        assert np.all(mask == 1.0)

    def test_revealing_own_words_zeroes_the_tail_of_the_own_mask_block(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        own_words = BOARD_WORDS[:9]
        # Reveal 3 of the 9 own words -- 6 remain.
        board = make_board(revealed=own_words[:3])
        vec = build_features(board, "clueone", sims, turn_index=0)
        layout = FeatureLayout(spaces=SPACES)
        mask = vec[layout.mask_slice()]
        own_start, own_end = 0, ROLE_COUNTS[Role.OWN]
        own_mask = mask[own_start:own_end]
        assert own_mask.sum() == 6
        assert list(own_mask) == [1.0] * 6 + [0.0] * 3

    def test_padded_slots_hold_the_sentinel_in_every_space(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board(revealed=BOARD_WORDS[24:25])  # reveal the single assassin word
        vec = build_features(board, "clueone", sims, turn_index=0)
        layout = FeatureLayout(spaces=SPACES)
        assassin_start, assassin_end = ROLE_COUNTS[Role.OWN] + ROLE_COUNTS[Role.OPPONENT] + ROLE_COUNTS[Role.NEUTRAL], 25
        for space in SPACES:
            values = vec[layout.space_slice(space)]
            assert values[assassin_start:assassin_end][0] == SENTINEL


class TestBuildFeaturesBatch:
    def test_matches_per_clue_build_features_for_every_clue(self, tmp_path):
        tensor = base_tensor()
        rng = np.random.default_rng(1)
        tensor[:] = rng.random(tensor.shape)
        # Sprinkle in some missing vectors so the NaN-handling path is exercised too.
        tensor[0, 3, 0] = np.nan
        tensor[1, 24, 1] = np.nan
        sims = make_sims(tmp_path, tensor)

        board = make_board(revealed=["Board0", "Board9", "Board24"])
        batch = build_features_batch(board, sims, turn_index=3)
        assert batch.shape == (len(CLUE_WORDS), feature_dim(len(SPACES)))

        for i, clue in enumerate(CLUE_WORDS):
            expected = build_features(board, clue, sims, turn_index=3)
            np.testing.assert_allclose(batch[i], expected)

    def test_batch_shape_with_no_words_left_in_a_role(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        # Reveal the only assassin word -- exercises the "no words in this
        # role" branch of the batch builder.
        board = make_board(revealed=["Board24"])
        batch = build_features_batch(board, sims, turn_index=1)
        assert not np.isnan(batch).any()
        assert batch.shape == (len(CLUE_WORDS), feature_dim(len(SPACES)))


class TestMissingVectorHandling:
    def test_word_missing_a_vector_in_one_space_still_produces_a_full_vector(self, tmp_path):
        tensor = base_tensor()
        # Board0 (an OWN word) has no vector in space 'a' for this clue.
        tensor[CLUE_WORDS.index("clueone"), BOARD_WORDS.index("Board0"), SPACES.index("a")] = np.nan
        sims = make_sims(tmp_path, tensor)
        board = make_board()

        vec = build_features(board, "clueone", sims, turn_index=0)
        assert not np.isnan(vec).any()

        layout = FeatureLayout(spaces=SPACES)
        own_values_a = vec[layout.space_slice("a")][:9]
        # 8 real values (0.5) sorted first, then one sentinel for the
        # missing-in-'a' word -- the shared mask still reports all 9 own
        # words as valid, since the word itself is real and unrevealed.
        assert list(own_values_a).count(SENTINEL) == 1
        mask = vec[layout.mask_slice()]
        assert mask[:9].sum() == 9
