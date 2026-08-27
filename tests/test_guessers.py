from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.guessers.base import Guesser
from codenames.guessers.blend import BlendGuesser
from codenames.guessers.confidence_threshold import ConfidenceThresholdGuesser
from codenames.guessers.noisy import NoisyGuesser
from codenames.guessers.rank_based import RankBasedGuesser
from codenames.guessers.registry import DEFAULT_POOL_CONFIG, load_pool
from codenames.guessers.single_space import SingleSpaceGuesser
from codenames.similarity import SimilarityTensor

BOARD_WORDS = ["Apple", "Banana", "Car", "Doghouse"]
SPACES = ["a", "b"]


@pytest.fixture
def sims(tmp_path):
    # Apple: high in both spaces. Banana: high in 'a', missing in 'b'.
    # Car: missing in 'a', moderate in 'b'. Doghouse: low in both.
    tensor = np.array(
        [[
            [0.9, 0.8],   # Apple
            [0.7, np.nan],  # Banana
            [np.nan, 0.5],  # Car
            [0.1, 0.1],   # Doghouse
        ]],
        dtype=np.float16,
    )
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestSingleSpaceGuesser:
    def test_ranks_by_raw_similarity(self, sims):
        g = SingleSpaceGuesser(space="a")
        assert g.rank_candidates("clue", BOARD_WORDS, sims) == ["Apple", "Banana", "Doghouse", "Car"]

    def test_missing_vector_scores_negative_infinity(self, sims):
        g = SingleSpaceGuesser(space="a")
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Car"] == float("-inf")

    def test_missing_vector_ranks_last(self, sims):
        g = SingleSpaceGuesser(space="a")
        ranked = g.rank_candidates("clue", BOARD_WORDS, sims)
        assert ranked[-1] == "Car"


class TestBlendGuesser:
    def test_uniform_weights_average_available_spaces(self, sims):
        g = BlendGuesser(weights={"a": 1.0, "b": 1.0})
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Apple"] == pytest.approx(0.85, abs=1e-2)  # mean(0.9, 0.8)

    def test_renormalizes_over_available_spaces_when_one_missing(self, sims):
        g = BlendGuesser(weights={"a": 1.0, "b": 1.0})
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        # Banana only has space 'a' (0.7) -- should use 0.7, not treat
        # the missing space as 0 (which would silently drag the average down)
        assert scores["Banana"] == pytest.approx(0.7, abs=1e-2)
        assert scores["Car"] == pytest.approx(0.5, abs=1e-2)

    def test_missing_in_all_weighted_spaces_is_negative_infinity(self, sims):
        g = BlendGuesser(weights={"a": 1.0})  # only space 'a', which Car lacks
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Car"] == float("-inf")

    def test_weights_change_ranking(self, tmp_path):
        # X and Y are both present in both spaces (unlike Banana/Car above,
        # which are each present in only one -- renormalizing over a
        # single available space makes its weight cancel out entirely, so
        # that pair can never demonstrate a weight-driven ranking change).
        words = ["X", "Y"]
        tensor = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float16)  # X: a=0.9,b=0.1; Y: a=0.1,b=0.9
        np.save(tmp_path / "similarity_tensor.npy", tensor)
        (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
        (tmp_path / "board_vocab.json").write_text(json.dumps(words))
        (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["a", "b"], "shape": list(tensor.shape)}))
        local_sims = SimilarityTensor.load(cache_dir=tmp_path)

        heavy_a = BlendGuesser(weights={"a": 10.0, "b": 1.0})
        heavy_b = BlendGuesser(weights={"a": 1.0, "b": 10.0})
        assert heavy_a.rank_candidates("clue", words, local_sims)[0] == "X"
        assert heavy_b.rank_candidates("clue", words, local_sims)[0] == "Y"


class TestRankBasedGuesser:
    def test_uses_rank_not_raw_score(self, tmp_path):
        # A has a huge outlier score in space 'a' that dominates any raw
        # average, but loses on rank in both spaces to B. Raw-score
        # blending picks A first; rank-based should pick B first instead.
        words = ["A", "B", "C"]
        tensor = np.array([[[100.0, 0.0], [2.0, 10.0], [1.0, 9.0]]], dtype=np.float16)
        np.save(tmp_path / "similarity_tensor.npy", tensor)
        (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
        (tmp_path / "board_vocab.json").write_text(json.dumps(words))
        (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["a", "b"], "shape": list(tensor.shape)}))
        local_sims = SimilarityTensor.load(cache_dir=tmp_path)

        raw_blend = BlendGuesser(weights={"a": 1.0, "b": 1.0})
        rank_based = RankBasedGuesser(spaces=["a", "b"])

        assert raw_blend.rank_candidates("clue", words, local_sims)[0] == "A"
        assert rank_based.rank_candidates("clue", words, local_sims)[0] == "B"

    def test_missing_vector_excluded_from_rank_average(self, sims):
        g = RankBasedGuesser(spaces=["a", "b"])
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        # Banana: rank 2 in 'a' (Apple=0.9 first, Banana=0.7 second, Doghouse=0.1 third among valid-in-a)
        # only one valid rank contributes for Banana (missing in 'b')
        assert scores["Banana"] == pytest.approx(-2.0, abs=1e-6)

    def test_missing_in_all_spaces_is_negative_infinity(self, sims):
        g = RankBasedGuesser(spaces=[])
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert all(s == float("-inf") for s in scores.values())


class TestNoisyGuesser:
    def test_same_seed_is_reproducible(self, sims):
        base = SingleSpaceGuesser(space="a")
        g1 = NoisyGuesser(base=base, noise_std=0.5, seed=7)
        g2 = NoisyGuesser(base=base, noise_std=0.5, seed=7)
        assert g1.rank_candidates("clue", BOARD_WORDS, sims) == g2.rank_candidates("clue", BOARD_WORDS, sims)

    def test_negative_infinity_is_not_perturbed(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = NoisyGuesser(base=base, noise_std=5.0, seed=1)
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Car"] == float("-inf")

    def test_noise_actually_changes_scores(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = NoisyGuesser(base=base, noise_std=1.0, seed=1)
        base_scores = base.score_candidates("clue", BOARD_WORDS, sims)
        noisy_scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert noisy_scores["Apple"] != base_scores["Apple"]


class TestConfidenceThresholdGuesser:
    def test_truncates_below_threshold(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = ConfidenceThresholdGuesser(base=base, threshold=0.5)
        # 'a' scores: Apple=0.9, Banana=0.7, Doghouse=0.1, Car=-inf
        ranked = g.rank_candidates("clue", BOARD_WORDS, sims)
        assert ranked == ["Apple", "Banana"]

    def test_threshold_above_everything_returns_empty(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = ConfidenceThresholdGuesser(base=base, threshold=999.0)
        assert g.rank_candidates("clue", BOARD_WORDS, sims) == []

    def test_threshold_below_everything_returns_all_valid(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = ConfidenceThresholdGuesser(base=base, threshold=-1.0)
        ranked = g.rank_candidates("clue", BOARD_WORDS, sims)
        assert ranked == ["Apple", "Banana", "Doghouse"]  # Car still excluded (-inf < -1.0)


class TestGuesserIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Guesser()


class TestRegistry:
    def test_default_pool_config_loads(self):
        entries = load_pool(DEFAULT_POOL_CONFIG)
        assert len(entries) == 8

    def test_exactly_two_held_out(self):
        entries = load_pool(DEFAULT_POOL_CONFIG)
        held_out = [name for name, e in entries.items() if e.held_out]
        assert len(held_out) == 2

    def test_wrapper_guessers_resolve_base_by_name(self):
        entries = load_pool(DEFAULT_POOL_CONFIG)
        assert isinstance(entries["noisy_blend"].guesser, NoisyGuesser)
        assert isinstance(entries["noisy_blend"].guesser.base, BlendGuesser)
        assert isinstance(entries["cautious_glove"].guesser, ConfidenceThresholdGuesser)
        assert isinstance(entries["cautious_glove"].guesser.base, SingleSpaceGuesser)

    def test_unknown_base_reference_raises(self, tmp_path):
        config = {
            "guessers": [
                {"name": "orphan", "type": "confidence_threshold", "params": {"base": "does_not_exist", "threshold": 0.1}},
            ]
        }
        path = tmp_path / "bad_pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="does_not_exist"):
            load_pool(path)

    def test_duplicate_name_raises(self, tmp_path):
        config = {
            "guessers": [
                {"name": "dup", "type": "single_space", "params": {"space": "a"}},
                {"name": "dup", "type": "single_space", "params": {"space": "b"}},
            ]
        }
        path = tmp_path / "dup_pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="dup"):
            load_pool(path)

    def test_unknown_type_raises(self, tmp_path):
        config = {"guessers": [{"name": "x", "type": "not_a_real_type", "params": {}}]}
        path = tmp_path / "bad_type_pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="not_a_real_type"):
            load_pool(path)
