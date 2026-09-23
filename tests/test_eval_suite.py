from __future__ import annotations

from types import SimpleNamespace

import pytest

import codenames.eval_suite as eval_suite_module
from codenames.board import Role
from codenames.eval_suite import (
    EvalSuite,
    file_content_hash,
    load_eval_suite,
    missing_seeds,
    run_eval_suite,
    seating,
    spymaster_identity,
)
from codenames.game import TurnResult, TwoTeamGameResult, TwoTeamTurnResult
from codenames.llm_store import GameRecordStore


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


GUESSER = "anthropic:claude-haiku-4-5"


@pytest.fixture
def suite() -> EvalSuite:
    return EvalSuite(name="test_suite", board_seeds=tuple(range(5)), guesser=GUESSER)


class TestLoadEvalSuite:
    def test_loads_the_committed_config(self):
        s = load_eval_suite()
        assert s.name
        assert len(s.board_seeds) > 0
        assert s.guesser

    def test_suite_id_is_stable_across_loads(self):
        assert load_eval_suite().suite_id == load_eval_suite().suite_id

    def test_suite_id_changes_with_the_guesser(self):
        a = EvalSuite("s", (0,), "anthropic:claude-haiku-4-5")
        b = EvalSuite("s", (0,), "anthropic:claude-sonnet-5")
        c = EvalSuite("s", (0,), "anthropic:claude-sonnet-5:medium")
        assert len({a.suite_id, b.suite_id, c.suite_id}) == 3

    def test_suite_id_is_independent_of_board_seeds(self):
        # Extending a suite to more boards must not invalidate results
        # already recorded under the smaller board_seeds tuple.
        a = EvalSuite("s", (0, 1, 2), GUESSER)
        b = EvalSuite("s", (0, 1, 2, 3, 4), GUESSER)
        assert a.suite_id == b.suite_id

    def test_the_committed_suite_guesser_builds(self):
        from codenames.guessers.registry import build_guesser
        assert build_guesser(load_eval_suite().guesser).cache_model_id == "claude-sonnet-5+effort=medium"


class TestSpymasterIdentity:
    def test_a_spymaster_with_nothing_to_hash_is_just_its_name(self):
        assert spymaster_identity("centroid") == "centroid"

    def test_parameters_are_part_of_the_identity(self):
        assert spymaster_identity("m", {"sigma": 1.5}) != spymaster_identity("m", {"sigma": 2.5})
        assert spymaster_identity("m", {"a": 1, "b": 2}) == spymaster_identity("m", {"b": 2, "a": 1})

    def test_a_model_file_retrained_in_place_is_a_different_model(self, tmp_path):
        path = tmp_path / "listener.txt"
        path.write_bytes(b"booster version one")
        id_a = spymaster_identity("learned", {}, [path])
        path.write_bytes(b"booster version two, totally different model")
        assert spymaster_identity("learned", {}, [path]) != id_a

    def test_same_bytes_give_the_same_identity_regardless_of_path(self, tmp_path):
        path_1, path_2 = tmp_path / "a.txt", tmp_path / "b.txt"
        path_1.write_bytes(b"identical bytes")
        path_2.write_bytes(b"identical bytes")
        assert spymaster_identity("learned", {}, [path_1]) == spymaster_identity("learned", {}, [path_2])
        assert file_content_hash(path_1) == file_content_hash(path_2)


class TestGameRecordStoreKeying:
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

    def test_label_only_calls_are_unaffected(self, tmp_path):
        # Sweeps record with a label and no suite key; those rows must
        # just append, never collide with each other.
        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(_by_role(), _result(seed=1), label="run-a")
        store.add_game(_by_role(), _result(seed=1), label="run-b")
        assert len(store.all_games()) == 2


IDS = ("challenger", "opponent")


def _record(store, suite, seed, a, b, winner="A"):
    result = _result(seed)
    result.winner = winner
    store.add_game(_by_role(), result, label=f"eval:{suite.name}|A={a},B={b}",
                   spymaster_id=seating(a, b), suite_id=suite.suite_id)


def _record_both(store, suite, seed):
    _record(store, suite, seed, *IDS)
    _record(store, suite, seed, IDS[1], IDS[0])


class TestRunEvalSuite:
    def test_a_board_counts_as_done_only_in_both_seatings(self, tmp_path, suite):
        db = tmp_path / "store.db"
        store = GameRecordStore(db)
        _record_both(store, suite, 0)
        _record(store, suite, 1, *IDS)            # one seating only
        assert missing_seeds(IDS, suite, db) == [1, 2, 3, 4]

    def test_rerun_of_a_fully_recorded_suite_plays_nothing(self, tmp_path, suite, monkeypatch):
        db = tmp_path / "store.db"
        store = GameRecordStore(db)
        for seed in suite.board_seeds:
            _record_both(store, suite, seed)

        def _boom(*args, **kwargs):
            raise AssertionError("nothing is missing, so nothing should be played")

        monkeypatch.setattr(eval_suite_module, "run_two_team_matchup", _boom)
        result = run_eval_suite(None, None, IDS, suite, game_record_db=db)
        assert result.played == 0
        assert result.paired.boards == len(suite.board_seeds)
        # Team A wins every game, so each board splits one-one.
        assert result.paired.split == len(suite.board_seeds)

    def test_extending_a_suite_only_plays_the_new_boards(self, tmp_path, monkeypatch):
        db = tmp_path / "store.db"
        store = GameRecordStore(db)
        small = EvalSuite("s", tuple(range(3)), GUESSER)
        for seed in small.board_seeds:
            _record_both(store, small, seed)
        big = EvalSuite("s", tuple(range(5)), GUESSER)
        assert big.suite_id == small.suite_id  # same identity, more boards

        captured = {}

        def _fake_matchup(spec_x, spec_y, names, *, seeds, game_record_db, suite_id, vocabulary, **kwargs):
            captured["seeds"] = list(seeds)
            captured["vocabulary"] = vocabulary
            rec = GameRecordStore(game_record_db)
            for seed in seeds:
                _record_both(rec, big, seed)
            return SimpleNamespace(discarded_boards=())

        monkeypatch.setattr(eval_suite_module, "run_two_team_matchup", _fake_matchup)
        result = run_eval_suite(None, None, IDS, big, game_record_db=db)
        assert captured["seeds"] == [3, 4]
        assert len(captured["vocabulary"]) == 150, "suite boards come from the held-out words"
        assert result.played == 2
        assert result.paired.boards == 5
