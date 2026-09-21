"""Fault tolerance and resumption for long arena sweeps.

Both behaviours here were written after a 7-hour, 11-setting sweep died 111
games in: gpt-oss declined to rank 19 words for the clue 'throat', the guesser
raised rather than fabricate a ranking (which is correct -- a made-up ranking
would be cached forever), and the exception propagated out of the pool and
ended every remaining setting.

The failure mode the resume path guards against is worse than a crash, because
it is silent: `GameRecordStore.add_game` keys on (spymaster_id, suite_id, seed)
and the arena passes neither, so re-running a setting APPENDS. Four rows per
seed means no board pairs, which means zero decisive boards, which means the
sign test quietly reports NaN rather than a wrong number or an error.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

import codenames.two_team_arena as arena

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_sweep", PROJECT_ROOT / "scripts" / "tools" / "sweep_role_costs.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

RUN = "testrun_opp=2"


def record(db: Path, seed: int, winner: str, a: str, b: str, run: str = RUN) -> None:
    """One game row in the shape analyze_headtohead/per_board parse."""
    con = sqlite3.connect(db)
    con.execute(
        "create table if not exists game_records "
        "(id integer primary key autoincrement, seed int, label text, board text, "
        " turns text, outcome text, winner text, total_reward real, "
        " spymaster_id text, suite_id text)")
    con.execute(
        "insert into game_records (seed, label, board, turns, outcome, winner, total_reward) "
        "values (?,?,?,?,?,?,?)",
        (seed, f"{run}|A={a},B={b}", "{}", json.dumps([]), "win", winner, 0.0))
    con.commit()
    con.close()


class TestUnpairedBoardsAreDropped:
    """A board played only one way must not reach any column of the table."""

    def test_a_half_played_board_counts_nowhere(self, tmp_path):
        db = tmp_path / "g.db"
        record(db, 1, "A", "base", "opp=2")            # paired below
        record(db, 1, "A", "opp=2", "base")
        record(db, 2, "A", "base", "opp=2")            # half only
        wins, _, (swept, split, boards) = sweep.per_board(db, RUN)
        assert boards == 1, "only the fully paired board counts"
        assert sum(wins.values()) == 2, "the orphan's win must not inflate win rate"
        assert swept.get("base", 0) + swept.get("opp=2", 0) + split == 1

    def test_win_rate_and_sign_test_describe_the_same_games(self, tmp_path):
        """The bug this prevents: an orphan inflating win rate while being
        absent from the pairing, so the two columns disagree."""
        db = tmp_path / "g.db"
        for seed in (1, 2):
            record(db, seed, "A", "base", "opp=2")
            record(db, seed, "B", "opp=2", "base")     # base wins both ways -> swept
        record(db, 99, "A", "opp=2", "base")           # orphan win for opp=2
        wins, _, (swept, split, boards) = sweep.per_board(db, RUN)
        assert boards == 2 and swept.get("base") == 2
        assert wins.get("opp=2", 0) == 0, "orphan must not appear as a win"
        assert sum(wins.values()) == 2 * boards


class TestResumeState:
    NAME = "opp=2"

    def test_absent_when_nothing_recorded(self, tmp_path):
        db = tmp_path / "g.db"
        record(db, 1, "A", "base", "x", run="other")
        assert sweep.resume_state(db, RUN, 2, {}, self.NAME) == "absent"

    def test_partial_when_rows_exist_but_the_setting_never_finished(self, tmp_path):
        db = tmp_path / "g.db"
        record(db, 1, "A", "base", "opp=2")
        record(db, 1, "B", "opp=2", "base")
        assert sweep.resume_state(db, RUN, 2, {}, self.NAME) == "partial"

    def test_complete_comes_from_the_progress_file_not_a_row_count(self, tmp_path):
        """The bug this replaces: a setting that finished WITH discarded boards
        writes fewer than 2*n_boards rows (ass=5 finished with 196 of 200), so
        counting rows called every real run partial and cleared it -- deleting
        precisely the work --resume exists to keep."""
        db = tmp_path / "g.db"
        for seed in (1, 2):                      # 4 rows, but n_boards says 100
            record(db, seed, "A", "base", "opp=2")
            record(db, seed, "B", "opp=2", "base")
        finished = {self.NAME: {"setting": self.NAME}}
        assert sweep.resume_state(db, RUN, 100, finished, self.NAME) == "complete"
        assert sweep.resume_state(db, RUN, 100, {}, self.NAME) == "partial"


class TestProgressFile:
    def test_round_trips_finished_settings(self, tmp_path):
        p = sweep.progress_path(tmp_path / "out.json", "lbl")
        sweep.save_progress(p, [{"setting": "ass=2", "win_rate": 0.569}])
        assert sweep.load_progress(p)["ass=2"]["win_rate"] == 0.569

    def test_a_missing_or_corrupt_file_resumes_from_scratch(self, tmp_path):
        """A half-written progress file must not stop the sweep from running."""
        assert sweep.load_progress(tmp_path / "nope.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert sweep.load_progress(bad) == {}


class TestClearRun:
    def test_removes_only_the_named_run(self, tmp_path):
        db = tmp_path / "g.db"
        record(db, 1, "A", "base", "opp=2")
        record(db, 1, "A", "base", "opp=3", run="testrun_opp=3")
        assert sweep.clear_run(db, RUN) == 1
        con = sqlite3.connect(db)
        left = [r[0] for r in con.execute("select label from game_records")]
        con.close()
        assert left == ["testrun_opp=3|A=base,B=opp=3"]

    def test_clearing_then_redoing_does_not_double_count(self, tmp_path):
        """Without the clear, per_board sees 4 rows per seed, pairs none of
        them, and the sign test silently has nothing to test."""
        db = tmp_path / "g.db"

        def play():
            for seed in (1, 2):
                record(db, seed, "A", "base", "opp=2")
                record(db, seed, "B", "opp=2", "base")

        play()
        play()                                          # the naive re-run
        _, _, (_, _, boards_appended) = sweep.per_board(db, RUN)
        assert boards_appended == 0, "4 rows per seed pair with nothing -- the silent failure"

        sweep.clear_run(db, RUN)
        play()
        _, _, (swept, _, boards) = sweep.per_board(db, RUN)
        assert boards == 2 and swept.get("base") == 2


class TestOneRefusalDoesNotEndTheMatchup:
    def test_matchup_task_returns_a_discard_instead_of_raising(self, monkeypatch):
        """The guesser's refusal arrives as RuntimeError inside a pool worker;
        propagating it ends every remaining setting."""
        def boom(*a, **k):
            raise RuntimeError("did not return a usable ranking for clue 'throat'")

        monkeypatch.setattr(arena, "play_two_team_game", boom)
        monkeypatch.setattr(arena, "_WORKER_STATE", {
            "sims": None, "x": object(), "y": object(), "names": ("base", "opp=2"),
            "guesser": object(), "max_turns": 40, "record_store": None, "run_label": "r",
        })
        result, swapped, seed, err = arena._matchup_task((4242, False))
        assert result is None and seed == 4242 and swapped is False
        assert "usable ranking" in err

    def test_an_unrelated_error_still_propagates(self, monkeypatch):
        """Only the guesser's documented refusal is swallowed. A bug in the
        game loop must not be silently recorded as a discarded board."""
        def boom(*a, **k):
            raise ValueError("a real bug")

        monkeypatch.setattr(arena, "play_two_team_game", boom)
        monkeypatch.setattr(arena, "_WORKER_STATE", {
            "sims": None, "x": object(), "y": object(), "names": ("base", "opp=2"),
            "guesser": object(), "max_turns": 40, "record_store": None, "run_label": "r",
        })
        with pytest.raises(ValueError):
            arena._matchup_task((4242, False))
