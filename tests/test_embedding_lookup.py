"""scripts/data/_embedding_lib.py: how a board word gets its vector, in
particular the abbreviation fallback ("St. patrick") that must not change any
word the name as written already resolves."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from _embedding_lib import lookup_tokens, resolve_board_vector  # noqa: E402

V = {w: np.full(2, float(i)) for i, w in enumerate(["st.", "saint", "patrick", "saint_patrick", "big", "ben"])}


def test_written_form_wins_when_it_resolves():
    # GloVe has "st.", so St. patrick keeps the mean of "st." and "patrick".
    vecs = {k: V[k] for k in ("st.", "saint", "patrick")}
    assert np.array_equal(resolve_board_vector("St. patrick", vecs, None), (V["st."] + V["patrick"]) / 2)


def test_abbreviation_spelled_out_only_as_a_fallback():
    # Numberbatch: no "st.", but a joined token for the spelled-out name.
    vecs = {k: V[k] for k in ("saint", "patrick", "saint_patrick")}
    assert np.array_equal(resolve_board_vector("St. patrick", vecs, "_"), V["saint_patrick"])
    # Wikipedia2Vec: no joined token, so the mean of the spelled-out parts.
    vecs = {k: V[k] for k in ("saint", "patrick")}
    assert np.array_equal(resolve_board_vector("St. patrick", vecs, None), (V["saint"] + V["patrick"]) / 2)


def test_plain_words_unchanged_and_missing_stays_missing():
    assert np.array_equal(resolve_board_vector("Big ben", V, None), (V["big"] + V["ben"]) / 2)
    assert resolve_board_vector("St. patrick", {"patrick": V["patrick"]}, None) is None


def test_lookup_tokens_covers_both_spellings():
    assert {"st.", "saint", "patrick", "st._patrick", "saint_patrick"} <= lookup_tokens("St. patrick", "_")
