from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.board import Board, Card, Role
from codenames.codemasters.base import MAX_CLUE_NUMBER, Codemaster
from codenames.codemasters.centroid import CentroidCodemaster
from codenames.codemasters.linear_scorer import DEFAULT_WEIGHTS, LinearScorerCodemaster
from codenames.codemasters.random_clue import RandomCodemaster
from codenames.similarity import SimilarityTensor

BOARD_WORDS = [f"Board{i}" for i in range(25)]
CLUE_WORDS = ["ownfavored", "opponentfavored", "neutralfavored", "assassinfavored", "mixedclue"]
SPACES = ["a", "b"]


def make_board(revealed: list[str] | None = None) -> Board:
    # 9 own, 8 opponent, 7 neutral, 1 assassin -- matches ROLE_COUNTS.
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(BOARD_WORDS, roles))
    board = Board(cards=cards, seed=1)
    for w in revealed or []:
        board.reveal(w)
    return board


def make_sims(tmp_path, tensor: np.ndarray) -> SimilarityTensor:
    np.save(tmp_path / "similarity_tensor.npy", tensor.astype(np.float16))
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


def base_tensor() -> np.ndarray:
    # (n_clues=5, n_board=25, n_spaces=2), all low similarity by default.
    return np.full((len(CLUE_WORDS), len(BOARD_WORDS), len(SPACES)), 0.05, dtype=np.float32)


class TestCodemasterIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Codemaster()


class TestRandomCodemaster:
    def test_returns_legal_clue_and_valid_number(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        cm = RandomCodemaster(seed=0)
        clue, number = cm.give_clue(board, sims)
        assert clue in sims.clue_words
        assert 1 <= number <= MAX_CLUE_NUMBER

    def test_deterministic_given_same_state(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        a = RandomCodemaster(seed=7).give_clue(board, sims)
        b = RandomCodemaster(seed=7).give_clue(board, sims)
        assert a == b

    def test_number_capped_by_own_remaining(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        # Reveal all but one own word -- number must be forced to 1.
        board = make_board(revealed=BOARD_WORDS[:8])
        cm = RandomCodemaster(seed=3)
        _, number = cm.give_clue(board, sims)
        assert number == 1


class TestCentroidCodemaster:
    def test_picks_clue_nearest_the_single_remaining_own_word(self, tmp_path):
        tensor = base_tensor()
        own0_idx = BOARD_WORDS.index("Board0")  # first OWN word
        ownfavored_idx = CLUE_WORDS.index("ownfavored")
        tensor[ownfavored_idx, own0_idx, :] = 0.95
        sims = make_sims(tmp_path, tensor)

        # Reveal every own word except Board0 -- forces a deterministic
        # single-word subset regardless of the sampling RNG.
        own_words = BOARD_WORDS[:9]
        board = make_board(revealed=[w for w in own_words if w != "Board0"])

        cm = CentroidCodemaster(seed=0)
        clue, number = cm.give_clue(board, sims)
        assert clue == "ownfavored"
        assert 1 <= number <= MAX_CLUE_NUMBER

    def test_deterministic_given_same_state(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        a = CentroidCodemaster(seed=5).give_clue(board, sims)
        b = CentroidCodemaster(seed=5).give_clue(board, sims)
        assert a == b


class TestLinearScorerCodemaster:
    def test_prefers_own_favored_over_assassin_favored(self, tmp_path):
        tensor = base_tensor()
        own_idxs = [BOARD_WORDS.index(f"Board{i}") for i in range(9)]
        opp_idxs = [BOARD_WORDS.index(f"Board{i}") for i in range(9, 17)]
        neutral_idxs = [BOARD_WORDS.index(f"Board{i}") for i in range(17, 24)]
        assassin_idx = BOARD_WORDS.index("Board24")

        tensor[CLUE_WORDS.index("ownfavored"), own_idxs, :] = 0.9
        tensor[CLUE_WORDS.index("assassinfavored"), assassin_idx, :] = 0.9
        tensor[CLUE_WORDS.index("opponentfavored"), opp_idxs, :] = 0.9
        tensor[CLUE_WORDS.index("neutralfavored"), neutral_idxs, :] = 0.9
        sims = make_sims(tmp_path, tensor)

        board = make_board()
        cm = LinearScorerCodemaster()
        clue, number = cm.give_clue(board, sims)
        assert clue == "ownfavored"
        assert 1 <= number <= MAX_CLUE_NUMBER

    def test_top_k_clues_ranked_best_first_and_agrees_with_give_clue(self, tmp_path):
        tensor = base_tensor()
        own_idxs = [BOARD_WORDS.index(f"Board{i}") for i in range(9)]
        assassin_idx = BOARD_WORDS.index("Board24")
        tensor[CLUE_WORDS.index("ownfavored"), own_idxs, :] = 0.9
        tensor[CLUE_WORDS.index("mixedclue"), own_idxs, :] = 0.5
        tensor[CLUE_WORDS.index("assassinfavored"), assassin_idx, :] = 0.9
        sims = make_sims(tmp_path, tensor)

        board = make_board()
        cm = LinearScorerCodemaster()
        top3 = cm.top_k_clues(board, sims, k=3)
        assert len(top3) == 3
        assert [c for c, _, _ in top3] == ["ownfavored", "mixedclue", "neutralfavored"]
        # scores strictly decreasing
        assert top3[0][2] > top3[1][2] > top3[2][2]

        clue, number = cm.give_clue(board, sims)
        assert (clue, number) == top3[0][:2]

    def test_default_weights_match_scope_baseline_3(self):
        assert DEFAULT_WEIGHTS[Role.OWN] == 1.0
        assert DEFAULT_WEIGHTS[Role.OPPONENT] == -1.0
        assert DEFAULT_WEIGHTS[Role.NEUTRAL] == -0.3
        assert DEFAULT_WEIGHTS[Role.ASSASSIN] == -10.0

    def test_gives_a_valid_clue_across_repeated_calls(self, tmp_path):
        # No per-instance tensor cache (see linear_scorer.py's docstring on
        # why) -- just check repeated calls keep working correctly.
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        cm = LinearScorerCodemaster()
        first = cm.give_clue(board, sims)
        second = cm.give_clue(board, sims)
        assert first == second
        assert first[0] in sims.clue_words
