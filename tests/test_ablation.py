from __future__ import annotations

import numpy as np
import pytest

from codenames.ablation import average_concatenation, drop_space
from codenames.features import FeatureLayout


class TestDropSpace:
    def test_removes_exactly_that_spaces_columns(self):
        layout = FeatureLayout(spaces=["a", "b", "c"])
        n = 4
        features = np.arange(n * layout.size, dtype=np.float32).reshape(n, layout.size)

        result = drop_space(features, layout, "b")

        assert result.shape == (n, layout.size - 25)
        np.testing.assert_array_equal(result[:, :25], features[:, layout.space_slice("a")])
        np.testing.assert_array_equal(result[:, 25:50], features[:, layout.space_slice("c")])
        np.testing.assert_array_equal(result[:, 50:75], features[:, layout.mask_slice()])
        np.testing.assert_array_equal(result[:, 75:78], features[:, layout.scalar_slice()])

    def test_dropping_different_spaces_gives_different_results(self):
        layout = FeatureLayout(spaces=["a", "b"])
        rng = np.random.default_rng(0)
        features = rng.random((3, layout.size)).astype(np.float32)
        drop_a = drop_space(features, layout, "a")
        drop_b = drop_space(features, layout, "b")
        assert drop_a.shape == drop_b.shape
        assert not np.allclose(drop_a, drop_b)


class TestAverageConcatenation:
    def test_positional_mean_of_space_blocks(self):
        layout = FeatureLayout(spaces=["a", "b"])
        features = np.zeros((1, layout.size), dtype=np.float32)
        features[0, layout.space_slice("a")] = 1.0
        features[0, layout.space_slice("b")] = 3.0
        features[0, layout.mask_slice()] = 1.0
        features[0, layout.scalar_slice()] = [5.0, 6.0, 7.0]

        result = average_concatenation(features, layout)

        assert result.shape == (1, 25 + 25 + 3)
        np.testing.assert_allclose(result[0, :25], 2.0)  # mean(1.0, 3.0)
        np.testing.assert_allclose(result[0, 25:50], 1.0)  # mask passed through
        np.testing.assert_allclose(result[0, 50:53], [5.0, 6.0, 7.0])  # scalars passed through

    def test_includes_sentinels_in_the_mean_without_special_casing(self):
        # Deliberately naive per the module docstring -- a -1 sentinel in
        # one space still gets averaged in like any other value.
        layout = FeatureLayout(spaces=["a", "b"])
        features = np.zeros((1, layout.size), dtype=np.float32)
        features[0, layout.space_slice("a")] = 0.8
        features[0, layout.space_slice("b")] = -1.0
        result = average_concatenation(features, layout)
        np.testing.assert_allclose(result[0, :25], -0.1)  # mean(0.8, -1.0)
