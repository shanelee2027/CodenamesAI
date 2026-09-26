"""The blind one-clue study in scripts/tools/play_server.py.

What these protect is the data, not the page. A study whose guesser could see
the arm, or the key, or skip the clues they disliked, would measure something
other than how well a spymaster's clues land with a human -- and nothing in the
recorded rows would show it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_play_server", PROJECT_ROOT / "scripts" / "tools" / "play_server.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

EvalTurn = ps.EvalTurn

#        own own own own opp neu ass
ROLES = ["a", "a", "a", "a", "b", "n", "x"]


def fresh(number=2, revealed=None):
    return EvalTurn(list(ROLES), revealed or [False] * len(ROLES), number)


class TestTurnRules:
    @pytest.mark.parametrize("i,ended", [(4, "opponent"), (5, "neutral"), (6, "assassin")])
    def test_a_wrong_pick_ends_the_turn(self, i, ended):
        t = fresh()
        t.pick(i)
        assert t.done and t.ended_by == ended

    def test_right_picks_continue_up_to_number_plus_one(self):
        t = fresh(number=2)
        t.pick(0); t.pick(1)
        assert not t.done, "the bonus guess is still available"
        t.pick(2)
        assert t.ended_by == "exhausted"

    def test_finding_every_own_word_ends_it(self):
        t = fresh(number=5, revealed=[True, True, False, False, False, False, False])
        t.pick(2); t.pick(3)
        assert t.ended_by == "cleared" and t.own_found() == 2

    def test_stop_is_a_recorded_outcome(self):
        """The human behaviour the outside option models; gpt-oss never does it."""
        t = fresh()
        t.pick(0)
        t.stop()
        assert t.ended_by == "stopped" and t.own_found() == 1

    def test_stopping_before_any_pick_is_allowed(self):
        t = fresh()
        t.stop()
        assert t.ended_by == "stopped" and t.own_found() == 0

    @pytest.mark.parametrize("setup,i", [("revealed", 0), ("twice", 1), ("out of range", 99)])
    def test_illegal_picks_are_refused(self, setup, i):
        t = fresh(revealed=[True] + [False] * 6)
        if setup == "twice":
            t.pick(1)
        with pytest.raises(ValueError):
            t.pick(i)

    def test_nothing_after_the_turn_ends(self):
        t = fresh()
        t.pick(5)
        with pytest.raises(ValueError):
            t.pick(0)
        with pytest.raises(ValueError):
            t.stop()


class FakeEngine:
    k1 = True

    def __init__(self):
        self.calls = []

    def best_clue(self, board, key, turn_index):
        self.calls.append((key, turn_index, set(board.revealed)))
        return "clue", 2, 1.5


@pytest.fixture
def study(tmp_path):
    return ps.EvalStudy(FakeEngine(), ["decoy", "decoy_out25"], tmp_path / "eval.jsonl")


class TestBlindness:
    def test_the_page_never_receives_the_arm_or_the_key(self, study):
        out = study.next("tester")
        assert set(out) == {"token", "words", "clue", "number", "revealed"}
        blob = json.dumps(out)
        assert "decoy" not in blob and "out25" not in blob
        hidden = [r for r in out["revealed"] if r is None]
        assert len(hidden) >= 25 - study.MAX_PREREVEAL, "only pre-revealed roles are sent"

    def test_arms_stay_balanced_within_one(self, study):
        counts = {"decoy": 0, "decoy_out25": 0}
        for n in range(1, 41):
            out = study.next("t")
            counts[study._pending[out["token"]]["arm"]] += 1
            assert abs(counts["decoy"] - counts["decoy_out25"]) <= 1, n

    def test_the_assassin_is_never_pre_revealed(self, study):
        for _ in range(60):
            out = study.next("t")
            assert "x" not in out["revealed"]

    def test_every_position_leaves_at_least_two_own_words(self, study):
        for _ in range(60):
            out = study.next("t")
            tok = out["token"]
            turn = study._pending[tok]["turn"]
            assert turn.own_left() >= 2


class TestPrefetch:
    def test_the_next_position_is_ready_before_it_is_asked_for(self, study):
        study.next("t")
        with study._ready_cv:                      # let the background compute land
            study._ready_cv.wait_for(lambda: study._ready, timeout=5)
        waiting = study._ready[0]
        out = study.next("t")
        served = study._pending[out["token"]]
        assert (served["seed"], served["arm"]) == (waiting["seed"], waiting["arm"])

    def test_time_shown_is_stamped_when_served_not_when_computed(self, study):
        study.next("t")
        with study._ready_cv:
            study._ready_cv.wait_for(lambda: study._ready, timeout=5)
        import time
        time.sleep(0.05)
        t = time.time()
        out = study.next("t")
        assert study._pending[out["token"]]["t_shown"] >= t

    def test_without_prefetch_nothing_runs_in_the_background(self, tmp_path):
        s = ps.EvalStudy(FakeEngine(), ["decoy", "decoy_out25"], tmp_path / "e.jsonl", prefetch=False)
        s.next("t")
        assert len(s.engine.calls) == 1 and not s._ready


class TestRecording:
    def test_a_finished_turn_writes_one_complete_row(self, study, tmp_path):
        out = study.next("alice")
        turn = study._pending[out["token"]]["turn"]
        own = next(i for i, r in enumerate(turn.roles) if r == "a" and not turn.revealed[i])
        study.pick(out["token"], own)
        res = study.stop(out["token"])
        assert res["done"] and res["recorded"] == 1
        row = json.loads((tmp_path / "eval.jsonl").read_text().strip())
        assert row["player"] == "alice" and row["arm"] in {"decoy", "decoy_out25"}
        assert row["ended_by"] == "stopped" and row["own_found"] == 1
        assert row["first_pick_own"] is True and row["stopped"] is True
        assert row["model"] == "listener_gbt_decoy.txt"
        assert row["picks"][0]["role"] == "a"

    def test_the_key_comes_back_only_once_the_turn_is_over(self, study):
        out = study.next("t")
        turn = study._pending[out["token"]]["turn"]
        own = next(i for i, r in enumerate(turn.roles) if r == "a" and not turn.revealed[i])
        mid = study.pick(out["token"], own)
        assert "key" not in mid, "the key must not leak mid-turn"

    def test_a_finished_position_cannot_be_replayed(self, study):
        out = study.next("t")
        study.stop(out["token"])
        with pytest.raises(ValueError):
            study.stop(out["token"])


class FakeCompareEngine:
    """Two spymasters that agree on the first `agree` boards, then differ."""
    k1 = True

    def __init__(self, agree=0):
        self.agree, self.boards = agree, 0

    def best_clue(self, board, key, turn_index):
        if key == "incumbent":
            self.boards += 1
        same = self.boards <= self.agree
        return ("clue" if same or key == "incumbent" else "other"), 2, 1.0

    def explain(self, board, key, clue, number):
        return {"targets": ["T1", "T2"], "value": 1.5,
                "order": [{"word": "T1", "role": "a", "p": 0.9}]}


class TestCompare:
    def test_agreeing_boards_are_dealt_past_and_counted(self, tmp_path):
        study = ps.CompareStudy(FakeCompareEngine(agree=3), tmp_path / "v.jsonl")
        out = study.next("incumbent", "assoc_pass", blind=False)
        assert out["agreed"] == 3
        assert {s["clue"] for s in out["sides"]} == {"clue", "other"}

    def test_blind_sends_only_clue_and_targets_until_the_vote(self, tmp_path):
        study = ps.CompareStudy(FakeCompareEngine(), tmp_path / "v.jsonl")
        out = study.next("incumbent", "assoc_pass", blind=True)
        assert all(set(s) == {"clue", "number", "targets"} for s in out["sides"])
        blob = json.dumps(out)
        assert "incumbent" not in blob and "assoc" not in blob
        res = study.vote(out["token"], "right", "why", "bob")
        assert {s["key"] for s in res["sides"]} == {"incumbent", "assoc_pass"}
        row = json.loads((tmp_path / "v.jsonl").read_text())
        assert row["winner"] == row["right"] and row["note"] == "why" and row["player"] == "bob"

    def test_a_board_takes_one_vote(self, tmp_path):
        study = ps.CompareStudy(FakeCompareEngine(), tmp_path / "v.jsonl")
        out = study.next("incumbent", "assoc_pass", blind=False)
        study.vote(out["token"], "tie", "")
        with pytest.raises(ValueError):
            study.vote(out["token"], "left", "")

    def test_same_model_twice_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            ps.CompareStudy(FakeCompareEngine(), tmp_path / "v.jsonl").next("decoy", "decoy", False)
