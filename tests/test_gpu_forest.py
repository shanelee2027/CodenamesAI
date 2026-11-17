"""GpuForest must reproduce LightGBM's raw predictions bit for bit (codenames/gpu_forest.py)."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lgb = pytest.importorskip("lightgbm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MODEL = Path(__file__).resolve().parents[1] / "cache" / "listener_gbt_fresh_split.txt"


def _toy_booster(rng):
    X = rng.normal(size=(3000, 6))
    X[rng.random(X.shape) < 0.1] = np.nan
    X[rng.random(X.shape) < 0.1] = 0.0
    y = np.nan_to_num(X[:, 0]) - np.nan_to_num(X[:, 1]) ** 2 + np.isnan(X[:, 2]) + rng.normal(size=3000) * 0.1
    params = {"objective": "regression", "num_leaves": 31, "verbosity": -1, "seed": 0,
              "use_missing": True, "zero_as_missing": False}
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=60), X


def _edge_rows(forest, X, rng):
    """Real rows with values set exactly ON thresholds, to NaN, to 0, and to +-tiny."""
    Z = X.copy()
    for f, ts in enumerate(forest.thresholds):
        if len(ts):
            m = rng.random(len(Z)) < 0.2
            Z[m, f] = rng.choice(ts, m.sum())
    Z[rng.random(Z.shape) < 0.05] = np.nan
    Z[rng.random(Z.shape) < 0.05] = 0.0
    Z[rng.random(Z.shape) < 0.02] = 1e-40
    Z[rng.random(Z.shape) < 0.02] = -1e-36
    return Z


def test_toy_model_matches_lightgbm():
    from codenames.gpu_forest import GpuForest

    rng = np.random.default_rng(0)
    booster, X = _toy_booster(rng)
    forest = GpuForest(booster, chunk_rows=700)
    for Z in (X, _edge_rows(forest, X, rng)):
        assert np.array_equal(forest.predict(Z), booster.predict(Z, raw_score=True))


@pytest.mark.skipif(not MODEL.exists(), reason="shipped listener not in cache/")
def test_shipped_listener_matches_lightgbm():
    from codenames.gpu_forest import GpuForest

    rng = np.random.default_rng(1)
    booster = lgb.Booster(model_file=str(MODEL))
    forest = GpuForest(booster, chunk_rows=1000)
    n = booster.num_feature()
    # Rows near real feature ranges: each column drawn around the thresholds the trees actually use.
    X = np.empty((4000, n))
    for f, ts in enumerate(forest.thresholds):
        ts = ts[np.abs(ts) < 1e6]                      # some splits sit at +-1e300
        lo, hi = (ts.min(), ts.max()) if len(ts) else (-1.0, 1.0)
        X[:, f] = rng.uniform(lo - 0.5, hi + 0.5, len(X))
    for Z in (X, _edge_rows(forest, X, rng)):
        assert np.array_equal(forest.predict(Z), booster.predict(Z, raw_score=True))
