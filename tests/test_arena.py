from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from codenames.arena import run_arena
from codenames.board import load_wordlist
from codenames.codemasters.centroid import CentroidCodemaster
from codenames.codemasters.random_clue import RandomCodemaster

CLUE_WORDS = ["clueone", "cluetwo", "cluethree"]
SPACES = ["a"]


@pytest.fixture
def sims_cache_dir(tmp_path):
    board_words = load_wordlist()  # real 400-word board vocab, so Board.generate finds matches
    tensor = np.full((len(CLUE_WORDS), len(board_words), len(SPACES)), 0.05, dtype=np.float16)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return tmp_path


@pytest.fixture
def guesser_pool_config(tmp_path):
    config = {
        "guessers": [
            {"name": "space_a", "type": "single_space", "params": {"space": "a"}},
            {"name": "space_a_holdout", "type": "single_space", "params": {"space": "a"}, "held_out": True},
        ]
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(config))
    return path


class TestRunArena:
    def test_plays_every_codemaster_against_every_guesser(self, sims_cache_dir, guesser_pool_config, tmp_path):
        codemaster_specs = {
            "random": (RandomCodemaster, {"seed": 0}),
            "centroid": (CentroidCodemaster, {"seed": 0}),
        }
        db_path = tmp_path / "arena.db"

        results, worker_rss = run_arena(
            codemaster_specs=codemaster_specs,
            guesser_pool_config=guesser_pool_config,
            seeds=[1, 2],
            db_path=db_path,
            sims_cache_dir=sims_cache_dir,
            max_turns=3,
            max_workers=1,
        )

        assert set(results.keys()) == {
            ("random", "space_a"),
            ("random", "space_a_holdout"),
            ("centroid", "space_a"),
            ("centroid", "space_a_holdout"),
        }
        for key, r in results.items():
            assert r.n_games == 2
            assert 0.0 <= r.win_rate <= 1.0
            assert 0.0 <= r.assassin_rate <= 1.0
            assert r.mean_turns > 0
            if r.mean_turns_on_win is not None:
                assert 0 < r.mean_turns_on_win <= r.mean_turns * r.n_games  # sane upper bound, not tight
            rates = (r.guess_own_rate, r.guess_opponent_rate, r.guess_neutral_rate, r.guess_assassin_rate)
            assert all(0.0 <= rate <= 1.0 for rate in rates)
            assert sum(rates) == pytest.approx(1.0)  # every guess lands on exactly one role

        assert results[("random", "space_a_holdout")].held_out is True
        assert results[("random", "space_a")].held_out is False
        assert len(worker_rss) >= 1

    def test_mean_turns_on_win_is_none_when_nothing_won(self, sims_cache_dir, guesser_pool_config, tmp_path):
        # max_turns=0 forces every game to time out (play_game's loop body
        # never runs) -- zero wins, so there's nothing to average.
        codemaster_specs = {"random": (RandomCodemaster, {"seed": 0})}
        db_path = tmp_path / "arena.db"

        results, _ = run_arena(
            codemaster_specs=codemaster_specs,
            guesser_pool_config=guesser_pool_config,
            seeds=[1, 2],
            db_path=db_path,
            sims_cache_dir=sims_cache_dir,
            max_turns=0,
            max_workers=1,
        )

        for r in results.values():
            assert r.win_rate == 0.0
            assert r.mean_turns_on_win is None

    def test_logs_one_row_per_turn_to_sqlite(self, sims_cache_dir, guesser_pool_config, tmp_path):
        codemaster_specs = {"random": (RandomCodemaster, {"seed": 0})}
        db_path = tmp_path / "arena.db"

        run_arena(
            codemaster_specs=codemaster_specs,
            guesser_pool_config=guesser_pool_config,
            seeds=[1],
            db_path=db_path,
            sims_cache_dir=sims_cache_dir,
            max_turns=3,
            max_workers=1,
        )

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT codemaster, guesser, board_seed, turn_index, clue, ended_reason, game_outcome FROM turns").fetchall()
        conn.close()

        assert len(rows) > 0
        for cm, guesser, seed, turn_index, clue, ended_reason, outcome in rows:
            assert cm == "random"
            assert guesser in ("space_a", "space_a_holdout")
            assert seed == 1
            assert isinstance(turn_index, int)
            assert clue in CLUE_WORDS
            assert ended_reason in ("own_words_complete", "opponent", "neutral", "assassin", "exhausted_guesses", "no_guesses")
            assert outcome in ("win", "loss", "timeout")
