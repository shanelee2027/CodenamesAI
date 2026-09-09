"""codenames/spymasters/z_threshold.py -- the z-scored threshold baseline.

Uses a synthetic ClueStats with mean=0, std=1 for every clue/space, which
makes the tensor's raw stored value *equal* to its z-score -- lets every
test set up exact, easy-to-reason-about z-scores directly, instead of
deriving them through build_clue_stats.py's mean/std computation (that
computation itself is covered by tests/test_clue_stats.py)."""

from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pytest

from codenames.board import Board, Card, Role
from codenames.clue_stats import ClueStats
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import MAX_CLUE_NUMBER, TurnContext
from codenames.spymasters.z_threshold import ZThresholdSpymaster

BOARD_WORDS = [f"Board{i}" for i in range(25)]
SPACE = "numberbatch"


def make_board(revealed: list[str] | None = None) -> Board:
    roles = [Role.OWN] * 9 + [Role.OPPONENT] * 8 + [Role.NEUTRAL] * 7 + [Role.ASSASSIN] * 1
    cards = tuple(Card(word=w, role=r) for w, r in zip(BOARD_WORDS, roles))
    board = Board(cards=cards, seed=1)
    for w in revealed or []:
        board.reveal(w)
    return board


def make_ctx(board: Board, turn_index: int | None = None) -> TurnContext:
    return TurnContext(board=board, turn_index=turn_index if turn_index is not None else len(board.revealed))


def make_sims(tmp_path, tensor: np.ndarray, clue_words: list[str]) -> SimilarityTensor:
    np.save(tmp_path / "similarity_tensor.npy", tensor.astype(np.float16))
    (tmp_path / "clue_vocab.json").write_text(json.dumps(clue_words))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": [SPACE], "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


def make_stats(clue_words: list[str], rarity: list[float] | None = None) -> ClueStats:
    n = len(clue_words)
    mean = np.zeros((n, 1), dtype=np.float32)
    std = np.ones((n, 1), dtype=np.float32)
    rarity_percentile = np.asarray(rarity, dtype=np.float32) if rarity is not None else np.zeros(n, dtype=np.float32)
    return ClueStats(mean=mean, std=std, rarity_percentile=rarity_percentile, clue_words=clue_words, spaces=[SPACE])


def make_row(own=None, opponent=None, neutral=None, assassin=None) -> list[float]:
    """One clue's (25,) row in BOARD_WORDS order, from per-role z lists.
    Any role list defaults to a safe, far-away -5.0 for every word of that
    role."""
    own = own if own is not None else [-5.0] * 9
    opponent = opponent if opponent is not None else [-5.0] * 8
    neutral = neutral if neutral is not None else [-5.0] * 7
    assassin = assassin if assassin is not None else [-5.0] * 1
    assert len(own) == 9 and len(opponent) == 8 and len(neutral) == 7 and len(assassin) == 1
    return own + opponent + neutral + assassin


def build(tmp_path, rows: dict[str, list[float]], rarity: dict[str, float] | None = None, **kwargs):
    clue_words = list(rows)
    tensor = np.asarray([rows[c] for c in clue_words], dtype=np.float32)[:, :, None]  # (n_clues, 25, 1)
    sims = make_sims(tmp_path, tensor, clue_words)
    rarity_list = [rarity.get(c, 0.0) for c in clue_words] if rarity else None
    stats = make_stats(clue_words, rarity_list)
    sm = ZThresholdSpymaster(space=SPACE, clue_stats=stats, **kwargs)
    return sm, sims


class TestThresholdsAtConstruction:
    def test_percentiles_convert_to_z_via_normaldist(self, tmp_path):
        sm, _ = build(tmp_path, {"c": make_row()})
        dist = NormalDist()
        assert sm.t_thresh == pytest.approx(dist.inv_cdf(1 - 0.10))
        assert sm.n_thresh == pytest.approx(dist.inv_cdf(1 - 0.10))
        assert sm.o_thresh == pytest.approx(dist.inv_cdf(1 - 0.20))
        assert sm.a_thresh == pytest.approx(dist.inv_cdf(1 - 0.30))


class TestThresholdFilteringAndNumberCap:
    def test_k_counts_only_own_words_above_t_and_caps_at_max_clue_number(self, tmp_path):
        # 6 own words clear the default own_top=0.10 threshold (t~1.28);
        # 3 don't. k must be capped at MAX_CLUE_NUMBER=4, not the raw 6.
        own = [3.0, 2.5, 2.2, 2.0, 1.9, 1.5, -1.0, -1.0, -1.0]
        sm, sims = build(tmp_path, {"clue": make_row(own=own)})
        board = make_board()
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue == "clue"
        assert number == MAX_CLUE_NUMBER == 4

    def test_number_reflects_true_count_when_below_the_cap(self, tmp_path):
        own = [2.0, 1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  # only 2 clear t
        sm, sims = build(tmp_path, {"clue": make_row(own=own)})
        board = make_board()
        _, number = sm.give_clue(make_ctx(board), sims)
        assert number == 2


class TestRevealedWordsExcluded:
    def test_revealed_own_words_dont_count_toward_k_or_shift_the_intended_set(self, tmp_path):
        own = [3.0, 2.5, 2.2, 2.0, 1.9, 1.5, -1.0, -1.0, -1.0]
        sm, sims = build(tmp_path, {"clue": make_row(own=own)})
        # Reveal the 3 highest-z own words -- 3 clear t among the rest
        # (2.0, 1.9, 1.5), not the original 6.
        board = make_board(revealed=["Board0", "Board1", "Board2"])
        _, number = sm.give_clue(make_ctx(board), sims)
        assert number == 3

    def test_revealed_distractor_cannot_veto_a_clue(self, tmp_path):
        # An opponent word above the threshold would normally make a clue
        # invalid -- but if that word is already revealed it must be
        # excluded from the check entirely.
        own = [2.0, 1.5] + [-1.0] * 7
        opponent = [5.0] + [-5.0] * 7  # Board9 is a dangerously high opponent word
        sm, sims = build(tmp_path, {"clue": make_row(own=own, opponent=opponent)})
        board = make_board(revealed=["Board9"])
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue == "clue"
        assert number == 2


class TestFallbacks:
    def test_fallback_one_fires_when_no_clue_is_fully_valid(self, tmp_path):
        # k=2 is eligible, but Board9 (opponent) sits above o_thresh
        # (~0.84), so this clue is never "valid." With only one clue in
        # the vocabulary, fallback 1 (score every eligible clue ignoring
        # role thresholds) must still produce a usable pick.
        own = [2.0, 1.5] + [-1.0] * 7
        opponent = [2.0] + [-5.0] * 7
        sm, sims = build(tmp_path, {"clue": make_row(own=own, opponent=opponent)})
        board = make_board()
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue == "clue"
        assert number == 2

    def test_fallback_two_fires_when_no_own_word_clears_t(self, tmp_path):
        # Every own word is below t (~1.28) -- k=0 for every candidate.
        # Fallback 2 must force the intended set to the single highest
        # own word (here, -0.5) and still announce number=1.
        own = [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3]
        sm, sims = build(tmp_path, {"clue": make_row(own=own)})
        board = make_board()
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue == "clue"
        assert number == 1

    def test_never_raises_with_no_own_words_left_unrevealed(self, tmp_path):
        sm, sims = build(tmp_path, {"clue": make_row()})
        board = make_board(revealed=[f"Board{i}" for i in range(9)])  # every own word gone
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue in sims.clue_words
        assert number >= 1


class TestDeterminism:
    def test_same_board_twice_gives_identical_clue(self, tmp_path):
        rows = {
            "cluea": make_row(own=[2.0, 1.8] + [-1.0] * 7),
            "clueb": make_row(own=[2.2, 1.6] + [-1.0] * 7),
            "cluec": make_row(own=[1.9, 1.7] + [-1.0] * 7),
        }
        sm, sims = build(tmp_path, rows)
        board = make_board()
        a = sm.give_clue(make_ctx(board), sims)
        b = ZThresholdSpymaster(space=SPACE, clue_stats=sm.clue_stats).give_clue(make_ctx(make_board()), sims)
        assert a == b


class TestReturnedClueIsAlwaysLegal:
    def test_an_illegal_top_candidate_is_skipped_for_a_legal_runner_up(self, tmp_path):
        # "board0" normalizes to the same string as board word "Board0"
        # -- illegal by codenames.board.is_legal_clue -- and is given a
        # strictly better score (higher weakest z, same k) than the
        # legal alternative, so picking it would be a bug, not a
        # coincidence of tie-breaking.
        rows = {
            "board0": make_row(own=[5.0, 5.0] + [-1.0] * 7),
            "legalclue": make_row(own=[2.0, 2.0] + [-1.0] * 7),
        }
        sm, sims = build(tmp_path, rows)
        board = make_board()
        clue, _ = sm.give_clue(make_ctx(board), sims)
        assert clue == "legalclue"


class TestAssassinPenalizedMoreThanNeutral:
    def test_equal_margin_assassin_scores_lower_than_neutral(self, tmp_path):
        own = [2.0, 1.5] + [-1.0] * 7  # k=2, weakest=1.5 for both clues
        assassin_row = make_row(own=own, assassin=[1.4])  # margin 0.1 from weakest
        neutral_row = make_row(own=own, neutral=[-5.0] * 6 + [1.4])  # same margin

        dir_a, dir_n = tmp_path / "a", tmp_path / "n"
        dir_a.mkdir()
        dir_n.mkdir()
        sm_a, sims_a = build(dir_a, {"assassinclue": assassin_row})
        sm_n, sims_n = build(dir_n, {"neutralclue": neutral_row})

        best_n_a, scores_a = sm_a.score_batch(sims_a, [make_ctx(make_board())])[0]
        best_n_n, scores_n = sm_n.score_batch(sims_n, [make_ctx(make_board())])[0]

        idx_a = sims_a.clue_index["assassinclue"]
        idx_n = sims_n.clue_index["neutralclue"]
        assert best_n_a[idx_a] == 2
        assert best_n_n[idx_n] == 2
        assert scores_a[idx_a] < scores_n[idx_n]


class TestRarityFiltering:
    def test_a_clue_over_max_rarity_loses_to_a_worse_but_common_one(self, tmp_path):
        rows = {
            "rarebutgreat": make_row(own=[5.0, 5.0] + [-1.0] * 7),
            "commonbutokay": make_row(own=[2.0, 1.5] + [-1.0] * 7),
        }
        rarity = {"rarebutgreat": 99.0, "commonbutokay": 5.0}
        sm, sims = build(tmp_path, rows, rarity=rarity, max_rarity=10.0)
        board = make_board()
        clue, _ = sm.give_clue(make_ctx(board), sims)
        assert clue == "commonbutokay"


class TestScoreBatchProtocol:
    def test_score_batch_matches_top_clues_for_a_single_context(self, tmp_path):
        rows = {
            "cluea": make_row(own=[2.0, 1.8] + [-1.0] * 7),
            "clueb": make_row(own=[2.2, 1.6] + [-1.0] * 7),
        }
        sm, sims = build(tmp_path, rows)
        ctx = make_ctx(make_board())
        (clue, number, score) = sm.top_clues(ctx, sims, 1)[0]
        best_n, scores = sm.score_batch(sims, [ctx])[0]
        idx = sims.clue_index[clue.lower()]
        assert int(best_n[idx]) == number
        assert float(scores[idx]) == pytest.approx(score)

    def test_to_device_is_a_no_op(self, tmp_path):
        sm, _ = build(tmp_path, {"clue": make_row()})
        sm.to_device("cuda")  # must not raise even without a real device
