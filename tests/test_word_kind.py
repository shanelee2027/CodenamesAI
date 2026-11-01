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


def test_appended_after_the_first_69():
    """Older boosters read their columns by name, so the block comes after
    every feature that existed before it (69), and stays put when a later
    block is appended (cached positions are completed by position)."""
    assert tuple(FEATURE_NAMES[69:69 + len(KIND_FEATURES)]) == KIND_FEATURES


def test_text_block_appended_last():
    from codenames.listener_features import TEXT_FEATURES

    assert tuple(FEATURE_NAMES[-len(TEXT_FEATURES):]) == TEXT_FEATURES
