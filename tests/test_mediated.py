"""The mediated (two-hop association) block: which associations count as
mediators, how the columns are read off the table, and where the block sits."""

import numpy as np

from codenames.assoc_profile import mediators
from codenames.listener_features import FEATURE_NAMES, MEDIATED_FEATURES, ExtraSims, mediated_columns


def test_mediators_need_two_lists_and_leave_out_the_clue():
    lists = [["romance", "love", "beach", "honeymooners"],
             ["love", "romance", "honeymoons", "sunset"],
             ["beach", "love", "honeymooners", "moon"]]
    # love in 3 lists, romance and beach in 2 (romance earlier on average);
    # sunset in 1; honeymooners and honeymoons are the clue; moon is inside it.
    assert mediators(lists, "honeymoon") == ["love", "romance", "beach"]
    assert mediators(lists, "honeymoon", max_n=1) == ["love"]
    assert mediators([], "honeymoon") == []


def test_a_word_repeated_in_one_list_counts_once():
    assert mediators([["love", "love"], ["sun"]], "beach") == []


def _table():
    sp = {s: np.array([[1.0, 2.0, np.nan]]) * (k + 1) for k, s in enumerate(("med_glove", "med_numberbatch",
                                                                             "med_wikipedia2vec"))}
    return ExtraSims(spaces=sp, row_of={7: 0}, board_pos={"heart": 0, "hawaii": 1, "camel": 2})


def test_columns_follow_the_candidates():
    f = mediated_columns(_table(), 7, ["Hawaii", "zebra", "Heart", "Camel"])
    assert f.shape == (4, len(MEDIATED_FEATURES))
    assert np.allclose(f[0], [2, 4, 6, 0]) and np.allclose(f[2], [1, 2, 3, -2])
    assert np.isnan(f[1]).all() and np.isnan(f[3]).all()          # no column, no value: no evidence


def test_a_clue_outside_the_table_is_all_nan():
    assert np.isnan(mediated_columns(_table(), 8, ["Heart", "Hawaii"])).all()


def test_appended_last():
    assert tuple(FEATURE_NAMES[-len(MEDIATED_FEATURES):]) == MEDIATED_FEATURES
    assert len(FEATURE_NAMES) == 75 + len(MEDIATED_FEATURES)
