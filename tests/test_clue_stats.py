"""codenames/clue_stats.py: cached per-clue mean/std similarity and the
z-scoring helper `expected_words.py` builds on, plus
scripts/data/build_clue_stats.py's pure array-building function."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "data"))
from build_clue_stats import build_clue_stats  # noqa: E402

from codenames.clue_stats import ClueStats
from codenames.rollouts import clue_vocab_fingerprint
from codenames.similarity import SimilarityTensor

CLUE_WORDS = ["ownfavored", "opponentfavored", "neutralfavored", "assassinfavored", "mixedclue", "flatclue"]
BOARD_WORDS = [f"Board{i}" for i in range(25)]
SPACES = ["numberbatch"]


def make_sims(cache_dir: Path, tensor: np.ndarray) -> SimilarityTensor:
    np.save(cache_dir / "similarity_tensor.npy", tensor.astype(np.float16))
    (cache_dir / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (cache_dir / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (cache_dir / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=cache_dir)


def base_tensor(rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        return np.full((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES)), 0.05, dtype=np.float32)
    return rng.random((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES))).astype(np.float32)


def write_clue_stats_cache(cache_dir: Path, sims: SimilarityTensor, *, hash_override: str | None = None) -> None:
    mean, std, rarity = build_clue_stats(sims)
    np.savez(cache_dir / "clue_stats.npz", mean=mean, std=std, rarity_percentile=rarity)
    meta = {
        "clue_vocab_hash": hash_override if hash_override is not None else clue_vocab_fingerprint(sims.clue_words),
        "spaces": sims.spaces,
        "n_clues": len(sims.clue_words),
        "n_board_words": len(sims.board_words),
    }
    (cache_dir / "clue_stats_meta.json").write_text(json.dumps(meta))


class TestBuildClueStats:
    def test_mean_and_std_match_numpy_over_all_board_words(self, tmp_path):
        rng = np.random.default_rng(0)
        tensor = base_tensor(rng)
        sims = make_sims(tmp_path, tensor)

        mean, std, rarity = build_clue_stats(sims)

        assert mean.shape == (len(CLUE_WORDS), len(SPACES))
        assert std.shape == (len(CLUE_WORDS), len(SPACES))
        assert rarity.shape == (len(CLUE_WORDS),)
        # The tensor was saved as fp16, so compare against what the tensor
        # actually holds (already lossy) rather than the fp32 source.
        expected_mean = np.nanmean(np.asarray(sims.tensor, dtype=np.float32), axis=1)
        expected_std = np.nanstd(np.asarray(sims.tensor, dtype=np.float32), axis=1)
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(std, expected_std, rtol=1e-3, atol=1e-3)

    def test_rarity_percentile_matches_clue_search(self, tmp_path):
        from codenames.clue_search import clue_rarity_percentile

        sims = make_sims(tmp_path, base_tensor())
        _, _, rarity = build_clue_stats(sims)
        expected = clue_rarity_percentile(sims.clue_words)
        np.testing.assert_allclose(rarity, [expected[w] for w in sims.clue_words])


class TestClueStatsLoad:
    def test_loads_matching_cache(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        write_clue_stats_cache(tmp_path, sims)

        stats = ClueStats.load(cache_dir=tmp_path)
        assert stats.mean.shape == (len(CLUE_WORDS), len(SPACES))
        assert stats.std.shape == (len(CLUE_WORDS), len(SPACES))
        assert stats.rarity_percentile.shape == (len(CLUE_WORDS),)
        assert stats.clue_words == CLUE_WORDS
        assert stats.spaces == SPACES

    def test_rejects_vocab_hash_mismatch(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        write_clue_stats_cache(tmp_path, sims, hash_override="not-the-real-hash")

        with pytest.raises(ValueError, match="clue vocabulary"):
            ClueStats.load(cache_dir=tmp_path)

    def test_rejects_shape_mismatch_even_if_hash_happens_to_match(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        real_hash = clue_vocab_fingerprint(sims.clue_words)
        # Arrays built for a different (smaller) clue count than the live
        # tensor, but with a (falsified) matching hash -- shape mismatch
        # must still be caught rather than silently misaligning rows.
        np.savez(
            tmp_path / "clue_stats.npz",
            mean=np.zeros((3, len(SPACES)), dtype=np.float32),
            std=np.ones((3, len(SPACES)), dtype=np.float32),
            rarity_percentile=np.zeros(3, dtype=np.float32),
        )
        meta = {"clue_vocab_hash": real_hash, "spaces": SPACES, "n_clues": 3, "n_board_words": len(BOARD_WORDS)}
        (tmp_path / "clue_stats_meta.json").write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="shape"):
            ClueStats.load(cache_dir=tmp_path)


class TestZForBoard:
    def test_matches_manual_zscore(self, tmp_path):
        rng = np.random.default_rng(1)
        tensor = base_tensor(rng)
        sims = make_sims(tmp_path, tensor)
        write_clue_stats_cache(tmp_path, sims)
        stats = ClueStats.load(cache_dir=tmp_path)

        subset = ["Board0", "Board5", "Board12"]
        z = stats.z_for_board(sims, subset, "numberbatch")
        assert z.shape == (len(CLUE_WORDS), len(subset))

        si = stats.space_index("numberbatch")
        for j, w in enumerate(subset):
            sim = sims.similarities_for_board("ownfavored", [w], space="numberbatch")[0]
            idx = sims.clue_index["ownfavored"]
            expected = (sim - stats.mean[idx, si]) / stats.std[idx, si]
            assert z[idx, j] == pytest.approx(expected, rel=1e-3)

    def test_zero_for_a_clue_at_exactly_its_own_mean(self, tmp_path):
        # A clue with identical similarity to every board word has std=0
        # and every z is (x - x) / 0 == nan -- exercised here just to
        # confirm it doesn't raise, since np.errstate(divide="ignore")
        # must actually be in effect.
        tensor = base_tensor()  # every entry is 0.05 -- std is exactly 0 everywhere
        sims = make_sims(tmp_path, tensor)
        write_clue_stats_cache(tmp_path, sims)
        stats = ClueStats.load(cache_dir=tmp_path)

        z = stats.z_for_board(sims, ["Board0"], "numberbatch")
        assert np.all(np.isnan(z))
