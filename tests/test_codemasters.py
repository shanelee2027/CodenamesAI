from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.board import Board, Card, Role
from codenames.codemasters.base import MAX_CLUE_NUMBER, Codemaster
from codenames.codemasters.centroid import CentroidCodemaster
from codenames.codemasters.linear_scorer import DEFAULT_WEIGHTS, LinearScorerCodemaster
from codenames.codemasters.oracle import OracleCodemaster
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


def _suppress_unused_clues(tensor: np.ndarray, used: list[str]) -> None:
    # Every other clue in the fixture defaults to a uniform 0.05
    # everywhere; a flat tie's stable sort happens to favor own words
    # (they're listed first in BOARD_WORDS' role order), which would
    # accidentally give unused clues a large run length. Rank an
    # opponent word first for every unused clue so it can never compete.
    for clue in CLUE_WORDS:
        if clue not in used:
            tensor[CLUE_WORDS.index(clue), BOARD_WORDS.index("Board9"), :] = 0.99


class TestOracleCodemaster:
    def test_picks_the_clue_with_the_longest_consecutive_own_run(self, tmp_path):
        tensor = base_tensor()
        clue_idx = CLUE_WORDS.index("ownfavored")
        # Top 5 by similarity are own words (descending, no ties), 6th is
        # an opponent word ranked just below them -- run length exactly 5.
        for i, value in enumerate([0.95, 0.90, 0.85, 0.80, 0.75]):
            tensor[clue_idx, BOARD_WORDS.index(f"Board{i}"), :] = value
        tensor[clue_idx, BOARD_WORDS.index("Board9"), :] = 0.70  # opponent, blocks the run
        _suppress_unused_clues(tensor, ["ownfavored"])
        sims = make_sims(tmp_path, tensor)

        board = make_board()
        cm = OracleCodemaster(space="a")
        clue, number = cm.give_clue(board, sims)
        assert clue == "ownfavored"
        assert number == 5  # number = the intended word count directly

    def test_top_k_reports_run_length_as_score_and_agrees_with_give_clue(self, tmp_path):
        tensor = base_tensor()
        # "ownfavored": run of 3. "mixedclue": run of 1.
        tensor[CLUE_WORDS.index("ownfavored"), BOARD_WORDS.index("Board0"), :] = 0.95
        tensor[CLUE_WORDS.index("ownfavored"), BOARD_WORDS.index("Board1"), :] = 0.90
        tensor[CLUE_WORDS.index("ownfavored"), BOARD_WORDS.index("Board2"), :] = 0.85
        tensor[CLUE_WORDS.index("ownfavored"), BOARD_WORDS.index("Board9"), :] = 0.10  # opponent, blocks
        tensor[CLUE_WORDS.index("mixedclue"), BOARD_WORDS.index("Board3"), :] = 0.95
        tensor[CLUE_WORDS.index("mixedclue"), BOARD_WORDS.index("Board9"), :] = 0.50  # opponent, blocks
        _suppress_unused_clues(tensor, ["ownfavored", "mixedclue"])
        sims = make_sims(tmp_path, tensor)

        board = make_board()
        cm = OracleCodemaster(space="a")
        top2 = cm.top_k_clues(board, sims, k=2)
        assert [c for c, _, _ in top2] == ["ownfavored", "mixedclue"]
        assert top2[0][1:] == (3, 3.0)  # number=run=3, score=run=3.0
        assert top2[1][1:] == (1, 1.0)

        clue, number = cm.give_clue(board, sims)
        assert (clue, number) == top2[0][:2]

    def test_zero_run_length_reports_number_zero(self, tmp_path):
        # Every clue's single highest-similarity word is an opponent word
        # -- the best achievable run length is 0 for all of them, so
        # number (== run length) is legitimately 0, not an error case.
        tensor = base_tensor()
        for clue in CLUE_WORDS:
            tensor[CLUE_WORDS.index(clue), BOARD_WORDS.index("Board9"), :] = 0.99
        sims = make_sims(tmp_path, tensor)
        board = make_board()
        cm = OracleCodemaster(space="a")
        _, number = cm.give_clue(board, sims)
        assert number == 0
