"""The is-a features (scripts/data/build_isa_sims.py) and how boosters read
feature columns by name (listener_features.booster_columns)."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from codenames.listener_features import FEATURE_NAMES, booster_columns, isa_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from build_isa_sims import ancestors, noun_synsets, path_score  # noqa: E402


def _booster(names: list[str]) -> lgb.Booster:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(names)))
    ds = lgb.Dataset(X, label=X[:, 0] > 0, feature_name=names)
    return lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, ds, num_boost_round=2)


def test_an_older_booster_reads_the_leading_columns() -> None:
    # Everything before the first appended block: what the incumbent was fitted on.
    names = FEATURE_NAMES[: FEATURE_NAMES.index("isa")]
    assert booster_columns(_booster(names)) == list(range(len(names)))


def test_a_block_subset_reads_its_own_columns() -> None:
    names = ["z_glove", "isa", "cmp_cw", "wn_wup"]
    assert booster_columns(_booster(names)) == [FEATURE_NAMES.index(n) for n in names]


def test_a_feature_extract_never_makes_is_refused() -> None:
    with pytest.raises(ValueError):
        booster_columns(_booster(["z_glove", "no_such_feature"]))


def test_isa_count() -> None:
    assert isa_count(np.array([0.0, 0.5, 0.25, 0.0])) == 2
    assert np.isnan(isa_count(np.array([np.nan, np.nan])))
    assert isa_count(np.array([np.nan, 0.0])) == 0


def test_path_score_is_directional_and_distance_scaled() -> None:
    pytest.importorskip("nltk")
    try:
        fruit = noun_synsets("fruit")
    except LookupError:
        pytest.skip("WordNet corpus not installed")
    assert path_score(ancestors("lemon"), fruit) > 0          # a lemon is a fruit
    assert path_score(ancestors("pie"), fruit) == 0           # a pie is not
    assert path_score(ancestors("fruit"), noun_synsets("lemon")) == 0   # not the other way round
    assert path_score(ancestors("car"), noun_synsets("automobile")) == 1.0   # shared synset
    assert np.isnan(path_score(ancestors("lemon"), noun_synsets("quickly")))  # no noun sense
    # Berlin reaches "country" only through its "area" sense, further up than China.
    country = noun_synsets("country")
    assert 0 < path_score(ancestors("berlin"), country) < path_score(ancestors("china"), country)
