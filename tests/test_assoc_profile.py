"""codenames/assoc_profile.py: the clue-profile and reverse-association
listener features, on a hand-built table."""

import numpy as np

from codenames.assoc_profile import B_FEATURES, PROFILE_FEATURES, AssocProfile, vagueness
from codenames.listener_features import FEATURE_NAMES


def _profile():
    b = np.arange(len(B_FEATURES), dtype=float)[None, :]
    lists = {
        "lion": [["cat", "king", "mane"], ["mane", "cats", "roar"]],    # names the clue once, once as a plural
        "car": [["road", "wheel"], ["engine", "road"]],                  # never names it
    }
    return AssocProfile({"cat": 0}, b, lists)


def test_columns_by_hand():
    f = _profile().columns("Cat", ["Lion", "car", "zebra"])
    assert f.shape == (3, len(PROFILE_FEATURES))
    assert np.allclose(f[:, :len(B_FEATURES)], np.arange(len(B_FEATURES)))     # B is a board constant
    share, rank, board_rank = f[:, -3], f[:, -2], f[:, -1]
    assert share[0] == 1.0 and rank[0] == (1 / 1 + 1 / 2) / 2                  # "cat" first, "cats" second
    assert share[1] == 0.0 and rank[1] == 0.0
    assert np.isnan(share[2]) and np.isnan(rank[2]) and np.isnan(board_rank[2])  # no lists: no evidence
    assert board_rank[0] == 1 and board_rank[1] == 2


def test_unknown_clue_has_no_profile_but_keeps_reverse_associations():
    f = _profile().columns("mane", ["lion"])
    assert np.isnan(f[0, :len(B_FEATURES)]).all()
    assert f[0, -3] == 1.0 and np.isclose(f[0, -2], (1 / 3 + 1 / 1) / 2)        # third, then first


def test_vagueness():
    distinct, overlap, first = vagueness([["a", "b"], ["a", "c"]])
    assert distinct == 3 / 4 and overlap == 1 / 3 and first == 1.0
    assert all(np.isnan(v) for v in vagueness([]))


def test_appended_last():
    """Older boosters read their columns by name, so the new features must
    come after every existing one."""
    assert tuple(FEATURE_NAMES[-len(PROFILE_FEATURES):]) == PROFILE_FEATURES
