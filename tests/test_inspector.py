from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from inspector import BASELINE_ROLE_WEIGHTS, baseline_score  # noqa: E402

from codenames.board import Board, Role
from codenames.similarity import SimilarityTensor


@pytest.fixture
def board():
    return Board.generate(seed=0)


@pytest.fixture
def sims(tmp_path, board):
    # every board word gets a distinct, deterministic similarity to "clue"
    # in a single space, so the weighted role means are hand-checkable.
    clue_words = ["clue"]
    board_words = list(board.words)
    tensor = np.array([[[0.5] for _ in board_words]], dtype=np.float16).reshape(1, len(board_words), 1)
    for i in range(len(board_words)):
        tensor[0, i, 0] = 0.1 * (i + 1)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(clue_words))
    (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["space_a"], "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestBaselineScore:
    def test_weights_match_scope_section_6(self):
        assert BASELINE_ROLE_WEIGHTS[Role.OWN] == 1.0
        assert BASELINE_ROLE_WEIGHTS[Role.OPPONENT] == -1.0
        assert BASELINE_ROLE_WEIGHTS[Role.NEUTRAL] == -0.3
        assert BASELINE_ROLE_WEIGHTS[Role.ASSASSIN] == -10.0

    def test_role_means_only_average_that_roles_words(self, sims, board):
        _, role_means = baseline_score(sims, board, "clue")
        for role in Role:
            words = board.words_by_role(role)
            expected_values = [sims.similarity("clue", w, space="space_a") for w in words]
            assert role_means[role] == pytest.approx(np.mean(expected_values), abs=1e-3)

    def test_revealed_words_excluded_from_role_mean(self, sims, board):
        own_words = board.words_by_role(Role.OWN)
        board.reveal(own_words[0])
        _, role_means = baseline_score(sims, board, "clue")
        remaining = own_words[1:]
        expected = np.mean([sims.similarity("clue", w, space="space_a") for w in remaining])
        assert role_means[Role.OWN] == pytest.approx(expected, abs=1e-3)

    def test_fully_revealed_role_contributes_zero(self, sims, board):
        for w in board.words_by_role(Role.ASSASSIN):
            board.reveal(w)
        _, role_means = baseline_score(sims, board, "clue")
        assert role_means[Role.ASSASSIN] == 0.0

    def test_total_is_weighted_sum_of_role_means(self, sims, board):
        total, role_means = baseline_score(sims, board, "clue")
        expected = sum(BASELINE_ROLE_WEIGHTS[r] * role_means[r] for r in Role)
        assert total == pytest.approx(expected, abs=1e-6)

    def test_nan_similarity_excluded_from_role_mean(self, tmp_path, board):
        board_words = list(board.words)
        tensor = np.full((1, len(board_words), 1), 0.5, dtype=np.float16)
        tensor[0, 0, 0] = np.nan  # first board word (some role) is OOV for this clue
        np.save(tmp_path / "similarity_tensor.npy", tensor)
        (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
        (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
        (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["space_a"], "shape": list(tensor.shape)}))
        sims_with_nan = SimilarityTensor.load(cache_dir=tmp_path)

        _, role_means = baseline_score(sims_with_nan, board, "clue")
        role_of_first_word = board.role_of(board_words[0])
        # every non-NaN value is 0.5, so the mean should be exactly 0.5
        # regardless of role size, as long as the NaN entry was excluded
        # rather than treated as 0.
        assert role_means[role_of_first_word] == pytest.approx(0.5, abs=1e-3)
