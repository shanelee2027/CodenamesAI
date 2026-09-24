from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.board import Board, Card, Role
from codenames.spymasters.base import MAX_CLUE_NUMBER, Spymaster, TurnContext
from codenames.spymasters.centroid import CentroidSpymaster
from codenames.spymasters.expected_words import ExpectedWordsSpymaster
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
        assert set(entries) == {"centroid", "expected_words", "learned_listener", "association_listener",
                                "sonnet_spymaster"}

    def test_the_paid_spymaster_is_never_picked_up_by_role(self):
        """Every game sonnet_spymaster plays costs money, so no arena may
        include it without naming it."""
        from codenames.spymasters.registry import spymaster_names

        assert "sonnet_spymaster" not in spymaster_names("baseline", "exploration")

    def test_entries_build_the_expected_classes(self):
        from codenames.spymasters.registry import load_spymasters

        entries = load_spymasters()
        assert isinstance(entries["centroid"].build(), CentroidSpymaster)
        assert isinstance(entries["expected_words"].build(), ExpectedWordsSpymaster)

    def test_no_entry_is_marked_trained(self):
        """Nothing in the project trains any more -- every registered
        spymaster is constructible straight from its config params, with
        no checkpoint to supply."""
        from codenames.spymasters.registry import load_spymasters

        entries = load_spymasters()
        assert all(not e.trained for e in entries.values())

    def test_spec_is_a_class_and_kwargs_tuple(self):
        from codenames.spymasters.registry import load_spymasters

        entries = load_spymasters()
        cls, kwargs = entries["centroid"].spec
        assert cls is CentroidSpymaster
        assert kwargs == {"seed": 0}
        assert isinstance(cls(**kwargs), CentroidSpymaster)  # the spec alone is enough to construct one

    def test_spymaster_spec_merges_overrides_into_config_params(self, tmp_path):
        from codenames.spymasters.registry import spymaster_spec

        cls, kwargs = spymaster_spec("expected_words", sigma=1.25, max_rarity=5.0)
        assert cls is ExpectedWordsSpymaster
        assert kwargs["sigma"] == 1.25            # override wins
        assert kwargs["max_rarity"] == 5.0
        assert kwargs["space"] == "numberbatch"   # untouched config param survives

    def test_accepts_an_already_parsed_config_dict_not_just_a_path(self):
        from codenames.spymasters.registry import load_spymasters

        config = {"spymasters": [{"name": "r", "type": "centroid", "params": {"seed": 9}}]}
        entries = load_spymasters(config)
        assert list(entries) == ["r"]
        assert entries["r"].params == {"seed": 9}

    def test_duplicate_name_raises(self, tmp_path):
        from codenames.spymasters.registry import load_spymasters

        config = {
            "spymasters": [
                {"name": "dup", "type": "centroid", "params": {}},
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
