from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-batched two-team arena needs CUDA")

from codenames.board import Board, load_wordlist  # noqa: E402
from codenames.clue_stats import clue_vocab_fingerprint  # noqa: E402
from codenames.guessers.base import Guesser  # noqa: E402
from codenames.similarity import SimilarityTensor  # noqa: E402
from codenames.spymasters.expected_words import ExpectedWordsSpymaster  # noqa: E402
from codenames.two_team_arena import MIXED_GUESSER, run_two_team_self_play  # noqa: E402
from codenames.two_team_gpu_arena import _play_batch_group, run_two_team_self_play_gpu  # noqa: E402


class _SlowGuesser(Guesser):
    """Ignores similarity entirely and just sleeps -- for proving
    _play_batch_group's per-game guesser calls overlap instead of running
    one at a time (see codenames/two_team_gpu_arena.py's docstring)."""

    def __init__(self, delay: float):
        self.delay = delay

    def score_candidates(self, clue, candidate_words, sims):
        time.sleep(self.delay)
        return {w: 0.0 for w in candidate_words}

CLUE_WORDS = ["clueone", "cluetwo", "cluethree", "cluefour"]
SPACES = ["a"]


@pytest.fixture
def sims_cache_dir(tmp_path):
    board_words = load_wordlist()
    rng = np.random.default_rng(0)
    tensor = rng.random((len(CLUE_WORDS), len(board_words), len(SPACES))).astype(np.float16)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    # mean=0/std=1 so a stored value equals its z-score -- same synthetic
    # ClueStats trick tests/test_expected_words_spymaster.py uses, written
    # to disk here because the CPU path reconstructs the spymaster from
    # picklable kwargs inside a worker process and can't be handed an
    # in-memory object.
    n = len(CLUE_WORDS)
    np.savez(
        tmp_path / "clue_stats.npz",
        mean=np.zeros((n, len(SPACES)), dtype=np.float32),
        std=np.ones((n, len(SPACES)), dtype=np.float32),
        rarity_percentile=np.zeros(n, dtype=np.float32),
    )
    (tmp_path / "clue_stats_meta.json").write_text(
        json.dumps({"clue_vocab_hash": clue_vocab_fingerprint(CLUE_WORDS), "n_clues": n, "spaces": SPACES})
    )
    return tmp_path


@pytest.fixture
def guesser_pool_config(tmp_path):
    # Deterministic (no NoisyGuesser) on purpose -- same reasoning as
    # tests/test_gpu_arena.py: this test's whole point is an exact-match
    # comparison against the CPU path, and a stateful guesser's behavior
    # would depend on call order, which isn't guaranteed identical
    # between the two paths.
    config = {"guessers": [{"name": "space_a", "type": "single_space", "params": {"space": "a"}}]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def two_guesser_pool_config(tmp_path):
    config = {
        "guessers": [
            {"name": "space_a", "type": "single_space", "params": {"space": "a"}},
            {"name": "space_a_again", "type": "single_space", "params": {"space": "a"}},
        ]
    }
    path = tmp_path / "pool2.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def spymaster_kwargs(sims_cache_dir):
    """Constructor kwargs rather than an instance: the CPU arena builds a
    fresh spymaster inside each worker, so these must be picklable."""
    return {"space": SPACES[0], "cache_dir": sims_cache_dir}


class TestRunTwoTeamSelfPlayGpu:
    def test_matches_the_cpu_arena_exactly_for_deterministic_guessers(self, sims_cache_dir, guesser_pool_config, spymaster_kwargs):
        sims = SimilarityTensor.load(cache_dir=sims_cache_dir)
        seeds = list(range(1, 21))

        gpu_spymaster = ExpectedWordsSpymaster(**spymaster_kwargs)
        gpu_result = run_two_team_self_play_gpu(
            spymaster=gpu_spymaster,
            guesser_pool_config=guesser_pool_config,
            guesser_name="space_a",
            seeds=seeds,
            sims=sims,
            batch_size=7,
            max_turns=10,
            device=torch.device("cuda"),
        )

        cpu_result = run_two_team_self_play(
            ExpectedWordsSpymaster,
            spymaster_kwargs,
            guesser_pool_config,
            "space_a",
            seeds,
            sims_cache_dir=sims_cache_dir,
            max_turns=10,
            max_workers=1,
        )

        assert gpu_result.n_games == cpu_result.n_games
        assert gpu_result.assassin_rate == pytest.approx(cpu_result.assassin_rate)
        assert gpu_result.mean_half_turns_all == pytest.approx(cpu_result.mean_half_turns_all)
        assert gpu_result.mean_half_turns_clean_finish == (
            pytest.approx(cpu_result.mean_half_turns_clean_finish) if cpu_result.mean_half_turns_clean_finish is not None else None
        )
        assert gpu_result.guess_own_rate == pytest.approx(cpu_result.guess_own_rate)
        assert gpu_result.guess_opponent_rate == pytest.approx(cpu_result.guess_opponent_rate)
        assert gpu_result.guess_neutral_rate == pytest.approx(cpu_result.guess_neutral_rate)
        assert gpu_result.guess_assassin_rate == pytest.approx(cpu_result.guess_assassin_rate)

    def test_batch_size_does_not_change_the_result(self, sims_cache_dir, guesser_pool_config, spymaster_kwargs):
        sims = SimilarityTensor.load(cache_dir=sims_cache_dir)
        seeds = list(range(1, 15))

        results_by_batch = {}
        for batch_size in (1, 5, 20):
            spymaster = ExpectedWordsSpymaster(**spymaster_kwargs)
            results_by_batch[batch_size] = run_two_team_self_play_gpu(
                spymaster=spymaster,
                guesser_pool_config=guesser_pool_config,
                guesser_name="space_a",
                seeds=seeds,
                sims=sims,
                batch_size=batch_size,
                max_turns=10,
                device=torch.device("cuda"),
            )

        base = results_by_batch[1]
        for batch_size, r in results_by_batch.items():
            assert r.assassin_rate == pytest.approx(base.assassin_rate), batch_size
            assert r.guess_own_rate == pytest.approx(base.guess_own_rate), batch_size

    def test_mixed_guesser_matches_cpu_path(self, sims_cache_dir, two_guesser_pool_config, spymaster_kwargs):
        sims = SimilarityTensor.load(cache_dir=sims_cache_dir)
        seeds = list(range(1, 21))

        gpu_spymaster = ExpectedWordsSpymaster(**spymaster_kwargs)
        gpu_result = run_two_team_self_play_gpu(
            spymaster=gpu_spymaster,
            guesser_pool_config=two_guesser_pool_config,
            guesser_name=MIXED_GUESSER,
            seeds=seeds,
            sims=sims,
            batch_size=7,
            max_turns=10,
            device=torch.device("cuda"),
        )
        cpu_result = run_two_team_self_play(
            ExpectedWordsSpymaster,
            spymaster_kwargs,
            two_guesser_pool_config,
            MIXED_GUESSER,
            seeds,
            sims_cache_dir=sims_cache_dir,
            max_turns=10,
            max_workers=1,
        )
        assert gpu_result.n_games == cpu_result.n_games
        assert gpu_result.assassin_rate == pytest.approx(cpu_result.assassin_rate)
        assert gpu_result.mean_clue_number == pytest.approx(cpu_result.mean_clue_number)
        assert gpu_result.mean_correct_per_clue == pytest.approx(cpu_result.mean_correct_per_clue)

    def test_batch_guesser_calls_overlap_instead_of_serializing(self, sims_cache_dir, spymaster_kwargs):
        sims = SimilarityTensor.load(cache_dir=sims_cache_dir)
        spymaster = ExpectedWordsSpymaster(**spymaster_kwargs)
        device = torch.device("cuda")
        spymaster.to_device(device)

        n_games, delay = 6, 0.2
        boards = [Board.generate(seed=s) for s in range(n_games)]
        guesser = _SlowGuesser(delay=delay)
        guessers = {b.seed: guesser for b in boards}

        start = time.monotonic()
        _play_batch_group(spymaster, guessers, boards, sims, max_turns=1, device=device)
        elapsed = time.monotonic() - start

        # Serialized, even a single half-turn round would take
        # n_games * delay; concurrent, the whole two-round game (max_turns=1
        # is 2 half-turns) should stay well under that.
        assert elapsed < n_games * delay
