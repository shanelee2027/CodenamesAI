from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from generate_training_data import (  # noqa: E402
    CLUE_MIX,
    MAX_K,
    generate,
    sample_clue,
    sample_partial_board,
    simulate_natural_stop,
)

from codenames.board import Board, Card, Role, load_wordlist  # noqa: E402
from codenames.features import feature_dim  # noqa: E402
from codenames.game import ROLE_REWARD  # noqa: E402
from codenames.guessers.base import Guesser  # noqa: E402
from codenames.scorer import N_OUTCOME_CLASSES  # noqa: E402
from codenames.similarity import SimilarityTensor  # noqa: E402

CLUE_WORDS = ["clueone", "cluetwo", "cluethree"]
SPACES = ["a"]
VOCAB = load_wordlist()


@pytest.fixture
def sims_cache_dir(tmp_path):
    board_words = load_wordlist()
    tensor = np.full((len(CLUE_WORDS), len(board_words), len(SPACES)), 0.3, dtype=np.float16)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUE_WORDS))
    (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return tmp_path


@pytest.fixture
def guesser_pool_config(tmp_path):
    config = {
        "guessers": [
            {"name": "trainable", "type": "single_space", "params": {"space": "a"}},
            {"name": "held_out_one", "type": "single_space", "params": {"space": "a"}, "held_out": True},
        ]
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(config))
    return path


class TestSamplePartialBoard:
    def test_never_reveals_the_assassin(self):
        rng = __import__("random").Random(0)
        for _ in range(50):
            board, _ = sample_partial_board(rng, VOCAB)
            assassin = board.words_by_role(Role.ASSASSIN)[0]
            assert not board.is_revealed(assassin)

    def test_always_leaves_at_least_one_own_word_unrevealed(self):
        rng = __import__("random").Random(0)
        for _ in range(50):
            board, _ = sample_partial_board(rng, VOCAB)
            assert board.remaining(Role.OWN) >= 1

    def test_revealed_count_matches_turn_index_returned(self):
        rng = __import__("random").Random(1)
        board, revealed_count = sample_partial_board(rng, VOCAB)
        assert revealed_count == len(board.revealed)

    def test_boards_are_drawn_only_from_the_given_vocabulary(self):
        # generate()'s real default is load_training_wordlist() -- verify
        # the restriction is actually honored, using a small vocabulary
        # here so a violation would be easy to spot.
        rng = __import__("random").Random(2)
        small_vocab = VOCAB[:30]
        for _ in range(20):
            board, _ = sample_partial_board(rng, small_vocab)
            assert set(board.words) <= set(small_vocab)


class TestSampleClue:
    def test_returns_a_legal_clue_from_the_vocabulary(self, sims_cache_dir):
        sims = SimilarityTensor.load(sims_cache_dir)
        rng = __import__("random").Random(0)
        board = Board.generate(seed=1)
        for _ in range(20):
            clue = sample_clue(rng, sims, board)
            if clue is not None:
                assert clue in sims.clue_words

    def test_mix_fractions_sum_to_one(self):
        assert CLUE_MIX["subset_topk"] + CLUE_MIX["any_word_topk"] + CLUE_MIX["random"] == pytest.approx(1.0)


class ScriptedGuesser(Guesser):
    def __init__(self, preferred_order: list[str]):
        self.preferred_order = preferred_order

    def score_candidates(self, clue, candidate_words, sims):
        return {w: -self.preferred_order.index(w) if w in self.preferred_order else float("-inf") for w in candidate_words}

    def rank_candidates(self, clue, candidate_words, sims):
        candidates = set(candidate_words)
        return [w for w in self.preferred_order if w in candidates]


class TestSimulateNaturalStop:
    def _board(self, revealed=None):
        roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
        words = [f"Board{i}" for i in range(25)]
        board = Board(cards=tuple(Card(w, r) for w, r in zip(words, roles)), seed=1)
        for w in revealed or []:
            board.reveal(w)
        return board

    def test_does_not_mutate_the_board(self):
        board = self._board()
        guesser = ScriptedGuesser(["Board0", "Board1"])
        simulate_natural_stop(board, "clue", guesser, sims=None)
        assert board.revealed == set()

    def test_caps_k_at_max_k_even_with_more_own_words_available(self):
        board = self._board()
        guesser = ScriptedGuesser([f"Board{i}" for i in range(9)])  # all 9 own words
        k, cause, reward = simulate_natural_stop(board, "clue", guesser, sims=None, max_k=MAX_K)
        assert k == MAX_K
        assert cause is None
        assert reward == MAX_K * ROLE_REWARD[Role.OWN]

    def test_stops_and_scores_the_miss_on_first_non_own(self):
        board = self._board()
        guesser = ScriptedGuesser(["Board0", "Board9"])  # Board9 is OPPONENT
        k, cause, reward = simulate_natural_stop(board, "clue", guesser, sims=None)
        assert k == 1
        assert cause == Role.OPPONENT
        assert reward == ROLE_REWARD[Role.OWN] + ROLE_REWARD[Role.OPPONENT]

    def test_zero_k_when_first_guess_is_not_own(self):
        board = self._board()
        guesser = ScriptedGuesser(["Board24"])  # ASSASSIN
        k, cause, reward = simulate_natural_stop(board, "clue", guesser, sims=None)
        assert k == 0
        assert cause == Role.ASSASSIN
        assert reward == ROLE_REWARD[Role.ASSASSIN]

    def test_neutral_cause_is_recorded_too(self):
        board = self._board()
        guesser = ScriptedGuesser(["Board0", "Board17"])  # Board17 is NEUTRAL
        k, cause, reward = simulate_natural_stop(board, "clue", guesser, sims=None)
        assert k == 1
        assert cause == Role.NEUTRAL
        assert reward == ROLE_REWARD[Role.OWN] + ROLE_REWARD[Role.NEUTRAL]


class TestGenerate:
    def test_writes_shards_with_correct_shapes_and_dtypes(self, sims_cache_dir, guesser_pool_config, tmp_path):
        output_dir = tmp_path / "out"
        produced = generate(
            n_examples=25,
            shard_size=10,
            output_dir=output_dir,
            seed=0,
            guesser_pool_config=guesser_pool_config,
            sims_cache_dir=sims_cache_dir,
        )
        assert produced == 25

        feature_shards = sorted(output_dir.glob("features_*.npy"))
        assert len(feature_shards) == 3  # 10 + 10 + 5

        dim = feature_dim(len(SPACES))
        total = 0
        for path in feature_shards:
            features = np.load(path)
            idx = path.stem.split("_")[1]
            outcomes = np.load(output_dir / f"outcome_{idx}.npy")
            rewards = np.load(output_dir / f"reward_{idx}.npy")
            seeds = np.load(output_dir / f"seed_{idx}.npy")

            assert features.dtype == np.float32
            assert features.shape[1] == dim
            assert outcomes.dtype == np.int32
            assert rewards.dtype == np.float32
            assert seeds.dtype == np.int64
            assert len(outcomes) == len(features) == len(rewards) == len(seeds)
            assert np.all((outcomes >= 0) & (outcomes < N_OUTCOME_CLASSES))
            total += len(features)
        assert total == 25

    def test_default_board_vocabulary_is_the_training_wordlist(self, sims_cache_dir, guesser_pool_config, tmp_path, monkeypatch):
        import generate_training_data as gtd

        from codenames.board import load_training_wordlist

        seen_vocab = []
        original = gtd.sample_partial_board

        def spy(rng, vocabulary):
            seen_vocab.append(vocabulary)
            return original(rng, vocabulary)

        monkeypatch.setattr(gtd, "sample_partial_board", spy)
        gtd.generate(
            n_examples=5,
            shard_size=5,
            output_dir=tmp_path / "out",
            seed=0,
            guesser_pool_config=guesser_pool_config,
            sims_cache_dir=sims_cache_dir,
        )
        assert seen_vocab
        assert set(seen_vocab[0]) == set(load_training_wordlist())

    def test_never_samples_a_held_out_guesser(self, sims_cache_dir, guesser_pool_config, tmp_path, monkeypatch):
        import generate_training_data as gtd

        seen_guessers = []
        original = gtd.simulate_natural_stop

        def spy(board, clue, guesser, sims, max_k=MAX_K):
            seen_guessers.append(guesser)
            return original(board, clue, guesser, sims, max_k)

        monkeypatch.setattr(gtd, "simulate_natural_stop", spy)
        gtd.generate(
            n_examples=10,
            shard_size=10,
            output_dir=tmp_path / "out",
            seed=0,
            guesser_pool_config=guesser_pool_config,
            sims_cache_dir=sims_cache_dir,
        )
        from codenames.guessers.registry import load_pool

        held_out_guessers = {e.guesser for e in load_pool(guesser_pool_config).values() if e.held_out}
        assert not any(g in held_out_guessers for g in seen_guessers)

    def test_resumes_by_adding_new_shards_not_overwriting(self, sims_cache_dir, guesser_pool_config, tmp_path):
        output_dir = tmp_path / "out"
        generate(n_examples=5, shard_size=5, output_dir=output_dir, seed=0, guesser_pool_config=guesser_pool_config, sims_cache_dir=sims_cache_dir)
        assert len(list(output_dir.glob("features_*.npy"))) == 1

        generate(n_examples=5, shard_size=5, output_dir=output_dir, seed=1, guesser_pool_config=guesser_pool_config, sims_cache_dir=sims_cache_dir)
        assert len(list(output_dir.glob("features_*.npy"))) == 2
