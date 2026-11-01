"""codenames/word_kind.py: the per-word kind columns, on a hand-built table,
and their place at the end of the feature list."""

import numpy as np

from codenames.listener_features import FEATURE_NAMES
from codenames.word_kind import KIND_FEATURES, WordKind


def test_columns_follow_the_candidates_not_the_clue():
    k = WordKind({"homer": 0, "code": 1}, np.array([[1 / 3, 0.001, 0.04], [0.0, 0.02, 0.05]]))
    f = k.columns("poet", ["Code", "zebra", "Homer"])
    assert f.shape == (3, len(KIND_FEATURES))
    assert np.allclose(f[0], [0.0, 0.02, 0.05]) and np.allclose(f[2], [1 / 3, 0.001, 0.04])
    assert np.isnan(f[1]).all()                                    # not in the table: no evidence
    assert np.array_equal(f, k.columns("anything", ["Code", "zebra", "Homer"]), equal_nan=True)


def test_appended_last():
    """Older boosters read their columns by name, so the new block comes
    after every existing feature."""
    assert tuple(FEATURE_NAMES[-len(KIND_FEATURES):]) == KIND_FEATURES
