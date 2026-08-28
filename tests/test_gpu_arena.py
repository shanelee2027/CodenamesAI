from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-batched arena needs CUDA")

from codenames.arena import run_arena  # noqa: E402
from codenames.board import load_wordlist  # noqa: E402
from codenames.codemasters.learned import LearnedCodemaster  # noqa: E402
from codenames.features import feature_dim  # noqa: E402
from codenames.gpu_arena import run_arena_gpu  # noqa: E402
from codenames.scorer import Scorer  # noqa: E402
from codenames.similarity import SimilarityTensor  # noqa: E402

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
    return tmp_path


@pytest.fixture
def guesser_pool_config(tmp_path):
    # Deterministic (no NoisyGuesser) on purpose -- this test's whole point
    # is an exact-match comparison against run_arena's CPU path, and a
    # stateful noisy RNG's draw sequence depends on task-to-worker
    # scheduling, which isn't guaranteed identical between the two paths.
    config = {"guessers": [{"name": "space_a", "type": "single_space", "params": {"space": "a"}}]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def checkpoint_path(tmp_path, sims_cache_dir):
    dim = feature_dim(len(SPACES))
    model = Scorer(input_dim=dim)
    path = tmp_path / "scorer.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": dim}, path)
    return path


class TestRunArenaGpu:
    def test_matches_the_cpu_arena_exactly_for_deterministic_guessers(self, sims_cache_dir, guesser_pool_config, checkpoint_path, tmp_path):
        sims = SimilarityTensor.load(cache_dir=sims_cache_dir)
        seeds = list(range(1, 21))

        gpu_codemaster = LearnedCodemaster(checkpoint_path, device="cpu")
        gpu_results = run_arena_gpu(
            codemaster=gpu_codemaster,
            codemaster_name="learned",
            guesser_pool_config=guesser_pool_config,
            seeds=seeds,
            db_path=tmp_path / "arena_gpu.db",
            sims=sims,
            batch_size=7,
            max_turns=10,
            device=torch.device("cuda"),
        )

        cpu_results, _ = run_arena(
            codemaster_specs={"learned": (LearnedCodemaster, {"checkpoint_path": checkpoint_path, "device": "cpu"})},
            guesser_pool_config=guesser_pool_config,
            seeds=seeds,
            db_path=tmp_path / "arena_cpu.db",
            sims_cache_dir=sims_cache_dir,
            max_turns=10,
            max_workers=1,
        )

        gpu = gpu_results["space_a"]
        cpu = cpu_results[("learned", "space_a")]

        assert gpu.n_games == cpu.n_games
        assert gpu.win_rate == pytest.approx(cpu.win_rate)
        assert gpu.assassin_rate == pytest.approx(cpu.assassin_rate)
        assert gpu.mean_turns == pytest.approx(cpu.mean_turns)
        assert gpu.mean_turns_on_win == (pytest.approx(cpu.mean_turns_on_win) if cpu.mean_turns_on_win is not None else None)
        assert gpu.guess_own_rate == pytest.approx(cpu.guess_own_rate)
        assert gpu.guess_opponent_rate == pytest.approx(cpu.guess_opponent_rate)
        assert gpu.guess_neutral_rate == pytest.approx(cpu.guess_neutral_rate)
        assert gpu.guess_assassin_rate == pytest.approx(cpu.guess_assassin_rate)

    def test_batch_size_does_not_change_the_result(self, sims_cache_dir, guesser_pool_config, checkpoint_path, tmp_path):
        sims = SimilarityTensor.load(cache_dir=sims_cache_dir)
        seeds = list(range(1, 15))

        results_by_batch = {}
        for batch_size in (1, 5, 20):
            codemaster = LearnedCodemaster(checkpoint_path, device="cpu")
            results = run_arena_gpu(
                codemaster=codemaster,
                codemaster_name="learned",
                guesser_pool_config=guesser_pool_config,
                seeds=seeds,
                db_path=tmp_path / f"arena_{batch_size}.db",
                sims=sims,
                batch_size=batch_size,
                max_turns=10,
                device=torch.device("cuda"),
            )
            results_by_batch[batch_size] = results["space_a"]

        base = results_by_batch[1]
        for batch_size, r in results_by_batch.items():
            assert r.win_rate == pytest.approx(base.win_rate), batch_size
            assert r.guess_own_rate == pytest.approx(base.guess_own_rate), batch_size
