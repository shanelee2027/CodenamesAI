from __future__ import annotations

import json

import numpy as np
import pytest

from codenames.board import Role, load_wordlist
from codenames.codemasters.random_clue import RandomCodemaster
from codenames.game import TurnResult, TwoTeamGameResult, TwoTeamTurnResult
from codenames.two_team_arena import _new_stats, finalize_result, run_two_team_self_play, update_stats

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
    config = {"guessers": [{"name": "space_a", "type": "single_space", "params": {"space": "a"}}]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(config))
    return path


def make_result(outcome: str, guesses_a: list[tuple[str, Role]], guesses_b: list[tuple[str, Role]]) -> TwoTeamGameResult:
    return TwoTeamGameResult(
        seed=1,
        outcome=outcome,
        winner="A" if outcome == "win" else "B",
        turns=[
            TwoTeamTurnResult(team="A", turn=TurnResult(clue="c", number=len(guesses_a), guesses=guesses_a)),
            TwoTeamTurnResult(team="B", turn=TurnResult(clue="c", number=len(guesses_b), guesses=guesses_b)),
        ],
    )


class TestStatsBookkeeping:
    def test_clean_finish_counted_separately_from_assassin_ending(self):
        s = _new_stats()
        clean = make_result("win", [("w1", Role.OWN)], [("w2", Role.OWN)])
        assassin = make_result("loss", [("w3", Role.ASSASSIN)], [])
        update_stats(s, clean)
        update_stats(s, assassin)
        result = finalize_result(s)

        assert result.n_games == 2
        assert result.assassin_rate == pytest.approx(0.5)
        # both games contribute to "all"; only the clean one to "clean finish"
        assert result.mean_half_turns_all == pytest.approx((2 + 2) / 2)  # 2 turns each game
        assert result.mean_half_turns_clean_finish == pytest.approx(2.0)

    def test_clean_finish_is_none_when_every_game_hit_the_assassin(self):
        s = _new_stats()
        update_stats(s, make_result("loss", [("w1", Role.ASSASSIN)], []))
        result = finalize_result(s)
        assert result.mean_half_turns_clean_finish is None

    def test_guess_role_rates_are_pooled_across_both_teams(self):
        s = _new_stats()
        update_stats(
            s,
            make_result(
                "win",
                [("w1", Role.OWN), ("w2", Role.OPPONENT)],
                [("w3", Role.OWN), ("w4", Role.NEUTRAL)],
            ),
        )
        result = finalize_result(s)
        assert result.guess_own_rate == pytest.approx(0.5)
        assert result.guess_opponent_rate == pytest.approx(0.25)
        assert result.guess_neutral_rate == pytest.approx(0.25)
        assert result.guess_assassin_rate == pytest.approx(0.0)


class TestRunTwoTeamSelfPlay:
    def test_runs_real_games_end_to_end(self, sims_cache_dir, guesser_pool_config):
        result = run_two_team_self_play(
            RandomCodemaster,
            {"seed": 0},
            guesser_pool_config,
            "space_a",
            seeds=list(range(4)),
            sims_cache_dir=sims_cache_dir,
            max_turns=5,
            max_workers=2,
        )
        assert result.n_games == 4
        assert 0.0 <= result.assassin_rate <= 1.0
        assert result.mean_half_turns_all > 0
        rates = [result.guess_own_rate, result.guess_opponent_rate, result.guess_neutral_rate, result.guess_assassin_rate]
        assert all(0.0 <= r <= 1.0 for r in rates)
        assert sum(rates) == pytest.approx(1.0)
