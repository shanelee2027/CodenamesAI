from __future__ import annotations

import json

import pytest

import codenames.eval_suite as eval_suite_module
from codenames.board import Role
from codenames.eval_suite import (
    EvalSuite,
    checkpoint_content_hash,
    load_eval_suite,
    run_eval_suite,
    spymaster_identity,
)
from codenames.game import TurnResult, TwoTeamGameResult, TwoTeamTurnResult
from codenames.llm_store import GameRecordStore, board_by_role


class _FakeBoard:
    def __init__(self, by_role):
        self._by_role = by_role

    def words_by_role(self, role):
        return self._by_role[role]


def _by_role():
    return {Role.OWN: ["a"], Role.OPPONENT: ["b"], Role.NEUTRAL: ["c"], Role.ASSASSIN: ["d"]}


def _result(seed: int) -> TwoTeamGameResult:
    turn = TurnResult(clue="clue", number=1, guesses=[("a", Role.OWN)], ended_reason="own_words_complete")
    return TwoTeamGameResult(
        seed=seed,
        turns=[TwoTeamTurnResult(team="A", turn=turn)],
        outcome="win",
        winner="A",
    )


@pytest.fixture
def guesser_pool_config(tmp_path):
    config = {"guessers": [{"name": "llm", "type": "llm", "params": {"cache_path": "cache/llm_store.db"}}]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def suite(guesser_pool_config) -> EvalSuite:
    return EvalSuite(
        name="test_suite",
        board_seeds=tuple(range(5)),
        guesser_pool_config=guesser_pool_config,
        guesser_name="llm",
        llm_model="claude-haiku-4-5",
    )


class TestLoadEvalSuite:
    def test_loads_the_committed_config(self):
        s = load_eval_suite()
        assert s.name
        assert len(s.board_seeds) > 0
        assert s.llm_model
        assert s.guesser_name

    def test_suite_id_is_stable_across_loads(self):
        assert load_eval_suite().suite_id == load_eval_suite().suite_id

    def test_suite_id_changes_with_llm_model(self, guesser_pool_config):
        a = EvalSuite("s", (0,), guesser_pool_config, "llm", "claude-haiku-4-5")
        b = EvalSuite("s", (0,), guesser_pool_config, "llm", "claude-sonnet-5")
        assert a.suite_id != b.suite_id

    def test_suite_id_is_independent_of_board_seeds(self, guesser_pool_config):
        # Extending a suite to more boards must not invalidate results
        # already recorded under the smaller board_seeds tuple.
        a = EvalSuite("s", (0, 1, 2), guesser_pool_config, "llm", "claude-haiku-4-5")
        b = EvalSuite("s", (0, 1, 2, 3, 4), guesser_pool_config, "llm", "claude-haiku-4-5")
        assert a.suite_id == b.suite_id

    def test_suite_id_changes_with_guesser_pool_config_content(self, tmp_path):
        path_a = tmp_path / "pool.json"
        path_a.write_text(json.dumps({"guessers": [{"name": "llm", "type": "llm", "params": {}}]}))
        a = EvalSuite("s", (0,), path_a, "llm", "claude-haiku-4-5")
        suite_id_before = a.suite_id
        path_a.write_text(json.dumps({"guessers": [{"name": "llm", "type": "llm", "params": {"seed": 1}}]}))
        b = EvalSuite("s", (0,), path_a, "llm", "claude-haiku-4-5")
        assert suite_id_before != b.suite_id


class TestSpymasterIdentity:
    def test_baseline_identity_is_just_the_name(self):
        assert spymaster_identity("centroid") == "centroid"

    def test_two_checkpoints_at_the_same_path_get_different_identities(self, tmp_path):
        path = tmp_path / "scorer_best.pt"
        path.write_bytes(b"checkpoint version one")
        id_a = spymaster_identity("learned", path)

        path.write_bytes(b"checkpoint version two, totally different model")
        id_b = spymaster_identity("learned", path)

        assert id_a != id_b

    def test_same_checkpoint_bytes_give_the_same_identity_regardless_of_path(self, tmp_path):
        path_1 = tmp_path / "a.pt"
        path_2 = tmp_path / "b.pt"
        path_1.write_bytes(b"identical bytes")
        path_2.write_bytes(b"identical bytes")
        assert spymaster_identity("learned", path_1) == spymaster_identity("learned", path_2)

    def test_checkpoint_content_hash_changes_with_content(self, tmp_path):
        path = tmp_path / "scorer_best.pt"
        path.write_bytes(b"one")
        hash_a = checkpoint_content_hash(path)
        path.write_bytes(b"two")
        hash_b = checkpoint_content_hash(path)
        assert hash_a != hash_b


class TestGameRecordStoreKeying:
    def test_two_checkpoints_at_the_same_path_do_not_collide_in_the_store(self, tmp_path):
        checkpoint_path = tmp_path / "scorer_best.pt"
        db_path = tmp_path / "store.db"
        store = GameRecordStore(db_path)

        checkpoint_path.write_bytes(b"first training run")
        id_a = spymaster_identity("learned", checkpoint_path)
        store.add_game(_by_role(), _result(seed=1), spymaster_id=id_a, suite_id="suite-x")

        # A second training run overwrites the same path with a
        # different model, per scripts/pipeline/train_scorer.py's docstring.
        checkpoint_path.write_bytes(b"second training run, different weights")
        id_b = spymaster_identity("learned", checkpoint_path)
        store.add_game(_by_role(), _result(seed=1), spymaster_id=id_b, suite_id="suite-x")

        assert id_a != id_b
        assert store.recorded_seeds(id_a, "suite-x") == {1}
        assert store.recorded_seeds(id_b, "suite-x") == {1}
        assert len(store.games_for_suite(id_a, "suite-x")) == 1
        assert len(store.games_for_suite(id_b, "suite-x")) == 1

    def test_rerunning_the_same_key_replaces_rather_than_duplicates(self, tmp_path):
        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(_by_role(), _result(seed=1), spymaster_id="model", suite_id="suite")
        store.add_game(_by_role(), _result(seed=1), spymaster_id="model", suite_id="suite")
        assert len(store.games_for_suite("model", "suite")) == 1

    def test_recorded_seeds_is_scoped_to_the_exact_spymaster_and_suite(self, tmp_path):
        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(_by_role(), _result(seed=1), spymaster_id="model_a", suite_id="suite")
        store.add_game(_by_role(), _result(seed=2), spymaster_id="model_b", suite_id="suite")
        store.add_game(_by_role(), _result(seed=3), spymaster_id="model_a", suite_id="other_suite")
        assert store.recorded_seeds("model_a", "suite") == {1}

    def test_legacy_label_only_calls_are_unaffected(self, tmp_path):
        # Pre-step-6 calling convention (codenames/two_team_arena.py-style
        # training diagnostics): no spymaster_id/suite_id at all. Must
        # still just append, not collide with each other.
        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(_by_role(), _result(seed=1), label="run-a")
        store.add_game(_by_role(), _result(seed=1), label="run-b")
        assert len(store.all_games()) == 2


class TestRunEvalSuite:
    def test_rerun_of_a_fully_recorded_suite_makes_no_simulation_calls(self, tmp_path, suite):
        db_path = tmp_path / "store.db"
        store = GameRecordStore(db_path)
        for seed in suite.board_seeds:
            store.add_game(_by_role(), _result(seed), spymaster_id="model", suite_id=suite.suite_id)

        def _boom(*args, **kwargs):
            raise AssertionError("run_two_team_self_play_gpu should not be called when nothing is missing")

        original = eval_suite_module.run_two_team_self_play_gpu
        eval_suite_module.run_two_team_self_play_gpu = _boom
        try:
            result = run_eval_suite(spymaster=None, spymaster_id="model", suite=suite, sims=None, game_record_db=db_path)
        finally:
            eval_suite_module.run_two_team_self_play_gpu = original

        assert result.n_games == len(suite.board_seeds)

    def test_extending_a_suite_only_simulates_the_new_seeds(self, tmp_path, guesser_pool_config):
        db_path = tmp_path / "store.db"
        store = GameRecordStore(db_path)
        small_suite = EvalSuite("s", tuple(range(3)), guesser_pool_config, "llm", "claude-haiku-4-5")
        for seed in small_suite.board_seeds:
            store.add_game(_by_role(), _result(seed), spymaster_id="model", suite_id=small_suite.suite_id)

        big_suite = EvalSuite("s", tuple(range(5)), guesser_pool_config, "llm", "claude-haiku-4-5")
        assert big_suite.suite_id == small_suite.suite_id  # same identity, more boards

        captured = {}

        # A minimal stand-in that records the newly "played" seeds into
        # the store, mirroring what the real function would do -- the
        # point of this test is which seeds get passed in, not the
        # (already-tested elsewhere) game-playing mechanics themselves.
        def _fake_run_and_record(spymaster, *, guesser_pool_config, guesser_name, seeds, sims, game_record_db,
                                  spymaster_id, suite_id, **kwargs):
            captured["seeds"] = list(seeds)
            recording_store = GameRecordStore(game_record_db)
            for seed in seeds:
                recording_store.add_game(_by_role(), _result(seed), spymaster_id=spymaster_id, suite_id=suite_id)

        original = eval_suite_module.run_two_team_self_play_gpu
        eval_suite_module.run_two_team_self_play_gpu = _fake_run_and_record
        try:
            result = run_eval_suite(
                spymaster=None, spymaster_id="model", suite=big_suite, sims=None, game_record_db=db_path
            )
        finally:
            eval_suite_module.run_two_team_self_play_gpu = original

        assert captured["seeds"] == [3, 4]
        assert result.n_games == 5
