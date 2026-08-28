from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_scorer import ShardedTrainingData, _is_val_seed, train  # noqa: E402

from codenames.scorer import N_K_CLASSES, LinearScorer, Scorer  # noqa: E402

FEATURE_DIM = 12


def _write_shard(data_dir: Path, index: int, n: int, rng: np.random.Generator) -> None:
    features = rng.random((n, FEATURE_DIM)).astype(np.float32)
    ks = rng.integers(0, N_K_CLASSES, size=n).astype(np.int32)
    rewards = rng.random(n).astype(np.float32)
    seeds = (np.arange(n) + index * n).astype(np.int64) % 1000  # spread across 0..999
    np.save(data_dir / f"features_{index:05d}.npy", features)
    np.save(data_dir / f"k_{index:05d}.npy", ks)
    np.save(data_dir / f"reward_{index:05d}.npy", rewards)
    np.save(data_dir / f"seed_{index:05d}.npy", seeds)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        _write_shard(tmp_path, i, 200, rng)
    return tmp_path


class TestIsValSeed:
    def test_splits_roughly_by_fraction(self):
        seeds = np.arange(10_000)
        is_val = _is_val_seed(seeds, val_fraction=0.1)
        assert 900 <= is_val.sum() <= 1100  # ~10% of 10,000

    def test_same_seed_always_lands_in_the_same_partition(self):
        seeds = np.array([5, 5, 5, 999, 999])
        is_val = _is_val_seed(seeds, val_fraction=0.2)
        assert is_val[0] == is_val[1] == is_val[2]
        assert is_val[3] == is_val[4]


class TestShardedTrainingData:
    def test_filters_rows_by_predicate(self, data_dir):
        train_ds = ShardedTrainingData(data_dir, lambda s: ~_is_val_seed(s, 0.1))
        val_ds = ShardedTrainingData(data_dir, lambda s: _is_val_seed(s, 0.1))
        assert len(train_ds) + len(val_ds) == 600
        assert len(val_ds) > 0
        assert len(train_ds) > 0

    def test_getitem_returns_tensor_and_int_label(self, data_dir):
        ds = ShardedTrainingData(data_dir, lambda s: np.ones_like(s, dtype=bool))
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (FEATURE_DIM,)
        assert isinstance(y, int)
        assert 0 <= y < N_K_CLASSES

    def test_feature_dim_matches_shard_width(self, data_dir):
        ds = ShardedTrainingData(data_dir, lambda s: np.ones_like(s, dtype=bool))
        assert ds.feature_dim == FEATURE_DIM


class TestTrain:
    def test_produces_checkpoint_and_diagnostic_files(self, data_dir, tmp_path):
        output_dir = tmp_path / "checkpoints"
        checkpoint_path = train(
            data_dir=data_dir,
            output_dir=output_dir,
            val_fraction=0.1,
            batch_size=32,
            max_epochs=2,
            patience=2,
            seed=0,
        )

        assert checkpoint_path == output_dir / "scorer_best.pt"
        assert checkpoint_path.exists()
        assert (output_dir / "training_curves.csv").exists()
        assert (output_dir / "training_curves.png").exists()
        assert (output_dir / "reliability_diagrams.png").exists()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["input_dim"] == FEATURE_DIM
        model = Scorer(input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state"])  # should not raise

    def test_raises_when_no_validation_examples(self, data_dir, tmp_path):
        with pytest.raises(ValueError, match="no validation examples"):
            train(data_dir=data_dir, output_dir=tmp_path / "out", val_fraction=0.0, max_epochs=1)

    def test_model_factory_trains_a_different_architecture(self, data_dir, tmp_path):
        output_dir = tmp_path / "checkpoints"
        checkpoint_path = train(
            data_dir=data_dir,
            output_dir=output_dir,
            val_fraction=0.1,
            batch_size=32,
            max_epochs=2,
            patience=2,
            seed=0,
            model_factory=LinearScorer,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = LinearScorer(input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state"])  # should not raise
        with pytest.raises(RuntimeError):
            Scorer(input_dim=checkpoint["input_dim"]).load_state_dict(checkpoint["model_state"])
