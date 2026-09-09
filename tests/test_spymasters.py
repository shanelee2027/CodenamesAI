from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.board import Board, Card, Role
from codenames.spymasters.base import MAX_CLUE_NUMBER, Spymaster, TurnContext
from codenames.spymasters.centroid import CentroidSpymaster
from codenames.spymasters.learned import LearnedSpymaster
from codenames.spymasters.linear_scorer import DEFAULT_WEIGHTS, LinearScorerSpymaster
from codenames.spymasters.oracle import OracleSpymaster
from codenames.spymasters.random_clue import RandomSpymaster
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


def make_ctx(board: Board, turn_index: int | None = None) -> TurnContext:
    return TurnContext(board=board, turn_index=turn_index if turn_index is not None else len(board.revealed))


class TestSpymasterIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Spymaster()

    def test_subclass_must_implement_top_clues(self):
        class Incomplete(Spymaster):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_give_clue_is_a_thin_wrapper_over_top_clues(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()

        class Fixed(Spymaster):
            def top_clues(self, ctx, sims, k):
                return [("ownfavored", 3, 1.0)][:k]

        clue, number = Fixed().give_clue(make_ctx(board), sims)
        assert (clue, number) == ("ownfavored", 3)


class TestRegistry:
    def test_default_config_loads_every_baseline(self):
        from codenames.spymasters.registry import DEFAULT_SPYMASTER_CONFIG, load_spymasters

        entries = load_spymasters(DEFAULT_SPYMASTER_CONFIG)
        assert set(entries) == {"random", "centroid", "linear_scorer", "oracle", "learned", "expected_words"}

    def test_entries_build_the_expected_classes(self):
        from codenames.spymasters.registry import load_spymasters

        entries = load_spymasters()
        assert isinstance(entries["random"].build(), RandomSpymaster)
        assert isinstance(entries["centroid"].build(), CentroidSpymaster)
        assert isinstance(entries["linear_scorer"].build(), LinearScorerSpymaster)
        assert isinstance(entries["oracle"].build(), OracleSpymaster)

    def test_only_learned_is_marked_trained(self):
        from codenames.spymasters.registry import load_spymasters

        entries = load_spymasters()
        assert entries["learned"].trained is True
        assert entries["random"].trained is False
        assert entries["centroid"].trained is False
        assert entries["linear_scorer"].trained is False
        assert entries["oracle"].trained is False
        assert entries["expected_words"].trained is False

    def test_spec_is_a_class_and_kwargs_tuple(self):
        from codenames.spymasters.registry import load_spymasters

        entries = load_spymasters()
        cls, kwargs = entries["centroid"].spec
        assert cls is CentroidSpymaster
        assert kwargs == {"seed": 0}
        assert isinstance(cls(**kwargs), CentroidSpymaster)  # the spec alone is enough to construct one

    def test_spymaster_spec_merges_overrides_into_config_params(self, tmp_path):
        from codenames.spymasters.registry import spymaster_spec

        cls, kwargs = spymaster_spec("learned", checkpoint_path=tmp_path / "x.pt", miss_penalty=-1.0)
        assert cls is LearnedSpymaster
        assert kwargs["checkpoint_path"] == tmp_path / "x.pt"
        assert kwargs["miss_penalty"] == -1.0

    def test_accepts_an_already_parsed_config_dict_not_just_a_path(self):
        from codenames.spymasters.registry import load_spymasters

        config = {"spymasters": [{"name": "r", "type": "random", "params": {"seed": 9}}]}
        entries = load_spymasters(config)
        assert list(entries) == ["r"]
        assert entries["r"].params == {"seed": 9}

    def test_duplicate_name_raises(self, tmp_path):
        from codenames.spymasters.registry import load_spymasters

        config = {
            "spymasters": [
                {"name": "dup", "type": "random", "params": {}},
                {"name": "dup", "type": "centroid", "params": {}},
            ]
        }
        path = tmp_path / "dup.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="dup"):
            load_spymasters(path)

    def test_unknown_type_raises(self, tmp_path):
        from codenames.spymasters.registry import load_spymasters

        config = {"spymasters": [{"name": "x", "type": "not_a_real_type", "params": {}}]}
        path = tmp_path / "bad_type.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="not_a_real_type"):
            load_spymasters(path)

    def test_unknown_name_raises_key_error(self):
        from codenames.spymasters.registry import spymaster_spec

        with pytest.raises(KeyError, match="nonexistent"):
            spymaster_spec("nonexistent")


class TestRandomSpymaster:
    def test_returns_legal_clue_and_valid_number(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        cm = RandomSpymaster(seed=0)
        clue, number = cm.give_clue(make_ctx(board), sims)
        assert clue in sims.clue_words
        assert 1 <= number <= MAX_CLUE_NUMBER

    def test_deterministic_given_same_state(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        a = RandomSpymaster(seed=7).give_clue(make_ctx(board), sims)
        b = RandomSpymaster(seed=7).give_clue(make_ctx(board), sims)
        assert a == b

    def test_number_capped_by_own_remaining(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        # Reveal all but one own word -- number must be forced to 1.
        board = make_board(revealed=BOARD_WORDS[:8])
        cm = RandomSpymaster(seed=3)
        _, number = cm.give_clue(make_ctx(board), sims)
        assert number == 1

    def test_top_clues_returns_k_distinct_legal_clues(self, tmp_path):
        # No real ranking exists for a random pick (see
        # spymasters/random_clue.py) -- top_clues just returns k
        # independent, distinct, legal random draws.
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        cm = RandomSpymaster(seed=1)
        top3 = cm.top_clues(make_ctx(board), sims, k=3)
        assert len(top3) == 3
        clues = [c for c, _, _ in top3]
        assert len(set(clues)) == 3  # distinct
        for clue, number, score in top3:
            assert clue in sims.clue_words
            assert 1 <= number <= MAX_CLUE_NUMBER
            assert score == 0.0

    def test_top_clues_k_1_matches_give_clue(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        cm = RandomSpymaster(seed=11)
        top1 = cm.top_clues(make_ctx(board), sims, k=1)
        clue, number = RandomSpymaster(seed=11).give_clue(make_ctx(board), sims)
        assert (clue, number) == top1[0][:2]


class TestCentroidSpymaster:
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

        cm = CentroidSpymaster(seed=0)
        clue, number = cm.give_clue(make_ctx(board), sims)
        assert clue == "ownfavored"
        assert 1 <= number <= MAX_CLUE_NUMBER

    def test_deterministic_given_same_state(self, tmp_path):
        sims = make_sims(tmp_path, base_tensor())
        board = make_board()
        a = CentroidSpymaster(seed=5).give_clue(make_ctx(board), sims)
        b = CentroidSpymaster(seed=5).give_clue(make_ctx(board), sims)
        assert a == b


class TestLinearScorerSpymaster:
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
        cm = LinearScorerSpymaster()
        clue, number = cm.give_clue(make_ctx(board), sims)
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
        cm = LinearScorerSpymaster()
        top3 = cm.top_clues(make_ctx(board), sims, k=3)
        assert len(top3) == 3
        assert [c for c, _, _ in top3] == ["ownfavored", "mixedclue", "neutralfavored"]
        # scores strictly decreasing
        assert top3[0][2] > top3[1][2] > top3[2][2]

        clue, number = cm.give_clue(make_ctx(board), sims)
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
        cm = LinearScorerSpymaster()
        first = cm.give_clue(make_ctx(board), sims)
        second = cm.give_clue(make_ctx(board), sims)
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


class TestOracleSpymaster:
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
        cm = OracleSpymaster(space="a")
        clue, number = cm.give_clue(make_ctx(board), sims)
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
        cm = OracleSpymaster(space="a")
        top2 = cm.top_clues(make_ctx(board), sims, k=2)
        assert [c for c, _, _ in top2] == ["ownfavored", "mixedclue"]
        assert top2[0][1:] == (3, 3.0)  # number=run=3, score=run=3.0
        assert top2[1][1:] == (1, 1.0)

        clue, number = cm.give_clue(make_ctx(board), sims)
        assert (clue, number) == top2[0][:2]

    def test_zero_run_length_floors_number_at_one(self, tmp_path):
        # Every clue's single highest-similarity word is an opponent word
        # -- the best achievable run length is 0 for all of them, but
        # number is floored at 1 like every other spymaster here.
        tensor = base_tensor()
        for clue in CLUE_WORDS:
            tensor[CLUE_WORDS.index(clue), BOARD_WORDS.index("Board9"), :] = 0.99
        sims = make_sims(tmp_path, tensor)
        board = make_board()
        cm = OracleSpymaster(space="a")
        _, number = cm.give_clue(make_ctx(board), sims)
        assert number == 1
