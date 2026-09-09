"""Rollout storage round-trip and encoding invariants
(codenames/rollouts.py, docs/iteration-architecture.md step 4)."""

from __future__ import annotations

import numpy as np
import pytest

from codenames.board import BOARD_SIZE, Board, OpponentBoardView, Role
from codenames.game import ROLE_REWARD
from codenames.rollouts import (
    CAUSE_NONE,
    ROLE_CODES,
    RolloutWriter,
    board_from_row,
    clue_vocab_fingerprint,
    load_manifest,
    load_rollouts,
    reward_for,
    shard_indices,
    write_manifest,
)

VOCAB = [f"word{i:03d}" for i in range(60)]


def _board(seed: int = 0) -> Board:
    return Board.generate(seed, VOCAB)


def _write_one(tmp_path, board, *, turn_index=3, k=2, cause=Role.NEUTRAL, swapped=False):
    writer = RolloutWriter(tmp_path, capacity=4)
    writer.add(
        board=board,
        board_word_index={w: i for i, w in enumerate(VOCAB)},
        clue_index=1234,
        guesser_index=1,
        turn_index=turn_index,
        k=k,
        cause=cause,
        board_seed=board.seed,
        swapped=swapped,
    )
    writer.flush(0)
    return load_rollouts(tmp_path, 0)


class TestRoundTrip:
    def test_board_state_survives_a_round_trip(self, tmp_path):
        board = _board(7)
        board.reveal(board.words[0])
        board.reveal(board.words[5])

        batch = _write_one(tmp_path, board)
        restored = board_from_row(batch, 0, VOCAB)

        assert list(restored.words) == list(board.words)
        assert {w: restored.role_of(w) for w in restored.words} == {w: board.role_of(w) for w in board.words}
        assert {w for w in restored.words if restored.is_revealed(w)} == {board.words[0], board.words[5]}
        assert restored.seed == board.seed

    def test_outcome_fields_survive(self, tmp_path):
        batch = _write_one(tmp_path, _board(1), turn_index=9, k=3, cause=Role.ASSASSIN)
        assert int(batch.turn_index[0]) == 9
        assert int(batch.k[0]) == 3
        assert int(batch.cause[0]) == ROLE_CODES[Role.ASSASSIN]

    def test_no_miss_uses_the_censored_sentinel(self, tmp_path):
        batch = _write_one(tmp_path, _board(1), k=4, cause=None)
        assert int(batch.cause[0]) == CAUSE_NONE
        # The sentinel must never collide with a real role code, or a
        # censored rollout would silently decode as a real miss.
        assert CAUSE_NONE not in set(ROLE_CODES.values())

    def test_swapped_perspective_is_stored_resolved(self, tmp_path):
        """An OpponentBoardView row must restore with the *view's* roles,
        so featurization and re-simulation need no perspective flag."""
        board = _board(3)
        view = OpponentBoardView(board)
        batch = _write_one(tmp_path, view, swapped=True)
        restored = board_from_row(batch, 0, VOCAB)

        assert bool(batch.swapped[0]) is True
        assert {w: restored.role_of(w) for w in restored.words} == {w: view.role_of(w) for w in view.words}
        # ...and that is genuinely different from the underlying board.
        assert any(restored.role_of(w) != board.role_of(w) for w in board.words)


class TestEncoding:
    def test_every_slot_is_representable(self, tmp_path):
        board = _board(11)
        for word in list(board.words)[:BOARD_SIZE]:
            board.reveal(word)
        batch = _write_one(tmp_path, board)
        restored = board_from_row(batch, 0, VOCAB)
        assert all(restored.is_revealed(w) for w in restored.words)

    def test_role_codes_are_pinned(self):
        """Pinned on purpose: reordering the Role enum must not silently
        reinterpret an already-written rollout set."""
        assert ROLE_CODES == {Role.OWN: 0, Role.OPPONENT: 1, Role.NEUTRAL: 2, Role.ASSASSIN: 3}

    def test_partial_flush_writes_only_filled_rows(self, tmp_path):
        writer = RolloutWriter(tmp_path, capacity=10)
        board = _board(2)
        for _ in range(3):
            writer.add(
                board=board,
                board_word_index={w: i for i, w in enumerate(VOCAB)},
                clue_index=0,
                guesser_index=0,
                turn_index=0,
                k=0,
                cause=Role.OPPONENT,
                board_seed=board.seed,
                swapped=False,
            )
        assert writer.flush(0) == 3
        assert len(load_rollouts(tmp_path, 0)) == 3

    def test_empty_flush_is_a_noop(self, tmp_path):
        assert RolloutWriter(tmp_path, capacity=4).flush(0) == 0
        assert shard_indices(tmp_path) == []


class TestDerivedReward:
    """Reward is derived, never stored -- see codenames/rollouts.py."""

    def test_matches_the_generator_formula(self):
        assert reward_for(2, Role.NEUTRAL, ROLE_REWARD) == pytest.approx(2 * 1.0 + -0.2)
        assert reward_for(0, Role.ASSASSIN, ROLE_REWARD) == pytest.approx(-10.0)
        assert reward_for(4, None, ROLE_REWARD) == pytest.approx(4.0)

    def test_reprices_without_touching_storage(self):
        """The point of deriving it: changing a reward constant reprices an
        existing rollout set for free, which docs/design-decisions.md
        requires (reward params live outside training)."""
        risk_averse = {**ROLE_REWARD, Role.ASSASSIN: -100.0}
        assert reward_for(1, Role.ASSASSIN, ROLE_REWARD) == pytest.approx(-9.0)
        assert reward_for(1, Role.ASSASSIN, risk_averse) == pytest.approx(-99.0)


class TestManifest:
    def test_fingerprint_detects_a_changed_clue_vocabulary(self):
        a = clue_vocab_fingerprint(["one", "two", "three"])
        assert a == clue_vocab_fingerprint(["one", "two", "three"])
        assert a != clue_vocab_fingerprint(["one", "two", "four"])
        assert a != clue_vocab_fingerprint(["one", "two"])

    def test_fingerprint_is_order_sensitive(self):
        """Clue indices are positional, so a reordered vocabulary is a
        different vocabulary even with identical membership."""
        assert clue_vocab_fingerprint(["a", "b"]) != clue_vocab_fingerprint(["b", "a"])

    def test_manifest_round_trips(self, tmp_path):
        write_manifest(
            tmp_path,
            board_vocab=VOCAB,
            clue_vocab_hash="deadbeef",
            n_clue_words=3,
            guesser_names=["g1", "g2"],
            guesser_pool_config="configs/guesser_pool.json",
            seed=17,
            extra={"swap_perspective_prob": 0.5},
        )
        manifest = load_manifest(tmp_path)
        assert manifest["board_vocab"] == VOCAB
        assert manifest["guesser_names"] == ["g1", "g2"]
        assert manifest["seed"] == 17
        assert manifest["swap_perspective_prob"] == 0.5

    def test_board_vocab_is_stored_in_full(self, tmp_path):
        """Stored rather than referenced so decoding never depends on a
        wordlist file that may have changed since (step 5 changes it)."""
        write_manifest(
            tmp_path,
            board_vocab=VOCAB,
            clue_vocab_hash="x",
            n_clue_words=1,
            guesser_names=[],
            guesser_pool_config="",
            seed=0,
        )
        batch = _write_one(tmp_path, _board(4))
        # Decoding uses the manifest's copy, not load_wordlist().
        restored = board_from_row(batch, 0, load_manifest(tmp_path)["board_vocab"])
        assert set(restored.words) <= set(VOCAB)


class TestStorageSize:
    def test_a_rollout_row_is_much_smaller_than_a_feature_row(self, tmp_path):
        """The measured ratio is ~4x at real scale (docs/log.md). This
        only guards the property, not the exact number."""
        writer = RolloutWriter(tmp_path, capacity=100)
        board = _board(5)
        for _ in range(100):
            writer.add(
                board=board,
                board_word_index={w: i for i, w in enumerate(VOCAB)},
                clue_index=7,
                guesser_index=0,
                turn_index=1,
                k=1,
                cause=Role.NEUTRAL,
                board_seed=board.seed,
                swapped=False,
            )
        writer.flush(0)
        rollout_bytes = sum(p.stat().st_size for p in tmp_path.glob("*.npy"))
        feature_bytes = 100 * 107 * np.dtype(np.float32).itemsize
        assert rollout_bytes < feature_bytes
