"""codenames/spymasters/expected_words.py -- the threshold-free baseline
that replaced z_threshold.py.

Uses a synthetic ClueStats with mean=0, std=1 for every clue/space, which
makes the tensor's raw stored value *equal* to its z-score -- lets every
test set up exact, easy-to-reason-about z-scores directly, instead of
deriving them through build_clue_stats.py's mean/std computation (covered
by tests/test_clue_stats.py). The pure-algebra tests
(TestGainAndPenaltyFormula) go one level lower still and call
`gain_and_penalty` directly with hand-picked numpy arrays -- no
SimilarityTensor/ClueStats fixture at all -- since that is where the
metric's actual math (the sub-linear gain term especially) lives."""

from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pytest

from codenames.board import Board, Card, Role
from codenames.clue_stats import ClueStats
from codenames.game import ROLE_REWARD
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import MAX_CLUE_NUMBER, TurnContext
from codenames.spymasters.expected_words import ExpectedWordsSpymaster, gain_and_penalty

BOARD_WORDS = [f"Board{i}" for i in range(25)]
SPACE = "numberbatch"
_PHI = NormalDist().cdf  # reference implementation of Phi for the pure-formula tests


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


_FAR_OWN = -20.0  # a weak own word, never intended to be part of k
_FAR_NON_OWN = -100.0  # a non-own word far enough below _FAR_OWN too (not just below a strong own
# word) that diff = b - a stays deeply negative even against the weakest own words a clue
# might still announce -- avoids an accidental diff==0 (q=0.5, not saturated) if both
# defaults were equal.


def make_row(own=None, opponent=None, neutral=None, assassin=None) -> list[float]:
    """One clue's (25,) row in BOARD_WORDS order, from per-role z lists.
    Any role list defaults to a safe, far-away value for every word of
    that role (far enough below any plausible own z that q_gain/q_pen
    saturate to ~0 at the default taus, so a role left at its default
    never contributes measurable gain or risk)."""
    own = own if own is not None else [_FAR_OWN] * 9
    opponent = opponent if opponent is not None else [_FAR_NON_OWN] * 8
    neutral = neutral if neutral is not None else [_FAR_NON_OWN] * 7
    assassin = assassin if assassin is not None else [_FAR_NON_OWN] * 1
    assert len(own) == 9 and len(opponent) == 8 and len(neutral) == 7 and len(assassin) == 1
    return own + opponent + neutral + assassin


def build(tmp_path, rows: dict[str, list[float]], rarity: dict[str, float] | None = None, **kwargs):
    clue_words = list(rows)
    tensor = np.asarray([rows[c] for c in clue_words], dtype=np.float32)[:, :, None]  # (n_clues, 25, 1)
    sims = make_sims(tmp_path, tensor, clue_words)
    rarity_list = [rarity.get(c, 0.0) for c in clue_words] if rarity else None
    stats = make_stats(clue_words, rarity_list)
    sm = ExpectedWordsSpymaster(space=SPACE, clue_stats=stats, **kwargs)
    return sm, sims


class TestGainAndPenaltyFormula:
    """Direct tests of the pure metric, bypassing the board/ClueStats
    machinery entirely -- one candidate, K_max=2, one non-own word."""

    def test_matches_a_monte_carlo_of_the_guesser_model(self):
        """The metric claims to be an exact expectation under 'the guesser
        perceives z_i + eps_i, eps ~ N(0, sigma), and works down its own
        order'. Simulate that guesser directly and check the claim, rather
        than re-encoding the formula as its own expected value -- which
        would only prove the code matches itself. The previous version of
        this test did exactly that, and locked in two errors: an
        independence approximation low by 0.37 expected words, and
        scoring P(the top j own words clear D) rather than P(N >= j)."""
        rng = np.random.default_rng(0)
        a = np.array([[3.0, 1.0]])
        b = np.array([[0.5, -0.4, 1.2]])
        costs = np.array([1.0, 0.2, 10.0])
        sigma = 1.8

        gain, penalty = gain_and_penalty(a, b, costs, sigma)

        # Tolerance is derived from the estimator's own standard error
        # rather than a fixed atol: the penalty estimator takes values up
        # to the assassin's cost of 10, so its variance is far higher than
        # the gain's, and a single hardcoded tolerance either flakes on one
        # or is vacuous for the other.
        n = 1_500_000
        own = a[0][None, :] + rng.normal(0, sigma, (n, a.shape[1]))
        dis = b[0][None, :] + rng.normal(0, sigma, (n, b.shape[1]))
        worst = dis.argmax(axis=1)
        D = dis.max(axis=1)
        # N counts EVERY own word the guesser reaches, not the top k by our
        # ordering: it has no idea which words we intended, so any own word
        # above D counts toward the k it is allowed.
        N = (own > D[:, None]).sum(axis=1)
        for k in (1, 2):
            revealed = np.minimum(k, N)
            gain_se = revealed.std() / np.sqrt(n)
            assert abs(gain[0, k - 1] - revealed.mean()) < 5 * gain_se + 1e-3
            # A miss is N < k, and costs whatever the guesser actually picks,
            # which is the strongest distractor -- not every distractor that
            # could have broken through.
            miss_cost = costs[worst] * (N < k)
            pen_se = miss_cost.std() / np.sqrt(n)
            assert abs(penalty[0, k - 1] - miss_cost.mean()) < 5 * pen_se + 1e-3

    def test_gain_is_sub_linear_in_k_when_a_distractor_is_nearby(self):
        # a_2's marginal contribution to gain is s_1 * s_2 <= s_1 <= 1,
        # strictly less than 1 whenever any distractor sits close enough
        # to either own word to give it nonzero q_gain. A linear metric
        # (gain(k) = k) would instead credit exactly +1 per k regardless
        # of the distractor -- the thing the spec explicitly forbids
        # "simplifying" to.
        a = np.array([[3.0, 1.0]])
        b = np.array([[0.5]])
        costs = np.array([1.0])
        gain, _ = gain_and_penalty(a, b, costs, sigma=1.8)
        marginal_second_word = gain[0, 1] - gain[0, 0]
        assert 0 < marginal_second_word < 1.0

    def test_gain_approaches_linear_only_once_distractors_are_far_enough(self):
        # With no distractor anywhere near either own word, s_j -> 1 for
        # every j, so gain(k) -> k -- the model reduces to "the linear
        # metric" only in the limit where there is no real risk, not as
        # a general simplification.
        a = np.array([[3.0, 1.0]])
        b = np.array([[-20.0]])
        costs = np.array([1.0])
        gain, penalty = gain_and_penalty(a, b, costs, sigma=1.8)
        np.testing.assert_allclose(gain, [[1.0, 2.0]], atol=1e-4)
        np.testing.assert_allclose(penalty, [[0.0, 0.0]], atol=1e-4)

    def test_far_away_distractor_contributes_almost_nothing(self):
        # Saturation property: q_w -> 0 as b_w falls far below a, for
        # *both* the gain and penalty terms, independent of how weak or
        # strong a itself is.
        a = np.array([[0.2, 0.1]])  # modest own z's, not extreme
        near = np.array([[0.15]])  # right next to a_2
        far = np.array([[-20.0]])
        costs = np.array([1.0])

        _, penalty_near = gain_and_penalty(a, near, costs, sigma=1.8)
        _, penalty_far = gain_and_penalty(a, far, costs, sigma=1.8)
        assert penalty_far[0, 1] < 1e-6
        assert penalty_near[0, 1] > penalty_far[0, 1] + 1e-3

    def test_assassin_costs_ten_times_opponent_at_equal_margin(self):
        # ROLE_REWARD makes an assassin exactly 10x an opponent
        # (-10.0 vs -1.0) -- verified against the real dict rather than a
        # hardcoded "10", so a future reward retune can't silently
        # desync this assertion from the code.
        a = np.array([[1.0]])
        b = np.array([[0.5]])
        cost_ratio = abs(ROLE_REWARD[Role.ASSASSIN]) / abs(ROLE_REWARD[Role.OPPONENT])
        _, penalty_assassin = gain_and_penalty(a, b, np.array([abs(ROLE_REWARD[Role.ASSASSIN])]), sigma=1.8)
        _, penalty_opponent = gain_and_penalty(a, b, np.array([abs(ROLE_REWARD[Role.OPPONENT])]), sigma=1.8)
        assert penalty_assassin[0, 0] / penalty_opponent[0, 0] == pytest.approx(cost_ratio)
        assert cost_ratio == pytest.approx(10.0)

    def test_assassin_costs_far_more_than_neutral_at_equal_margin(self):
        # Neutral's ROLE_REWARD magnitude (0.2) makes it the cheapest
        # distractor by a wide margin (50x cheaper than the assassin,
        # not merely 10x) -- this test documents the actual ratio rather
        # than assuming a round number.
        a = np.array([[1.0]])
        b = np.array([[0.5]])
        cost_ratio = abs(ROLE_REWARD[Role.ASSASSIN]) / abs(ROLE_REWARD[Role.NEUTRAL])
        _, penalty_assassin = gain_and_penalty(a, b, np.array([abs(ROLE_REWARD[Role.ASSASSIN])]), sigma=1.8)
        _, penalty_neutral = gain_and_penalty(a, b, np.array([abs(ROLE_REWARD[Role.NEUTRAL])]), sigma=1.8)
        assert penalty_assassin[0, 0] / penalty_neutral[0, 0] == pytest.approx(cost_ratio)
        assert cost_ratio == pytest.approx(50.0)


class TestKChosenJointlyWithClue:
    def test_joint_selection_picks_the_clue_and_k_with_the_highest_score(self, tmp_path):
        # Both clues face the same assassin at z=0.9. "risky" has a
        # strong first own word (3.0, margin 2.1 from the assassin) plus
        # a second (1.0) close enough to the assassin that k=2 is a bad
        # idea -- its best k is 1. "safe" has only one own word at all
        # (1.5, margin only 0.6 from the same assassin) -- its best k is
        # also 1, but at a much worse score since its only word is far
        # closer to the threat. The winner must be "risky" at k=1 -- not
        # "risky" at k=2 (a per-clue-then-fixed-k rule would get this
        # wrong, since a_2=1.0 alone looks fine) and not "safe".
        rows = {
            "risky": make_row(own=[3.0, 1.0] + [_FAR_OWN] * 7, assassin=[0.9]),
            "safe": make_row(own=[1.5] + [_FAR_OWN] * 8, assassin=[0.9]),
        }
        sm, sims = build(tmp_path, rows)
        clue, number = sm.give_clue(make_ctx(make_board()), sims)
        assert clue == "risky"
        assert number == 1

    def test_far_distractors_let_the_same_clue_prefer_a_higher_k(self, tmp_path):
        # Same own words as above, but every distractor is far from
        # everything -- with no risk anywhere, gain(k) is exactly k
        # (see TestGainAndPenaltyFormula's linear-limit test) and
        # penalty(k) is exactly 0 for every k, so the highest available
        # k (the MAX_CLUE_NUMBER cap) always wins. This is the mirror
        # image of the test above: whether k=1 or k=MAX_CLUE_NUMBER wins
        # for the *same* own words is decided entirely by what's nearby,
        # confirming k is chosen jointly with (in fact, driven by) risk,
        # not by the own words alone.
        rows = {"risky": make_row(own=[3.0, 1.0] + [_FAR_OWN] * 7)}
        sm, sims = build(tmp_path, rows)
        clue, number = sm.give_clue(make_ctx(make_board()), sims)
        assert clue == "risky"
        assert number == MAX_CLUE_NUMBER

    def test_number_never_exceeds_max_clue_number(self, tmp_path):
        own = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2]  # all 9 own words strong, all distractors far
        sm, sims = build(tmp_path, {"clue": make_row(own=own)})
        _, number = sm.give_clue(make_ctx(make_board()), sims)
        assert number == MAX_CLUE_NUMBER == 4


class TestRevealedWordsExcluded:
    def test_revealed_own_words_dont_count_toward_k(self, tmp_path):
        own = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2]
        sm, sims = build(tmp_path, {"clue": make_row(own=own)})
        # Only 6 own words remain unrevealed -- K_max is still capped at
        # MAX_CLUE_NUMBER, but z_for_board must never see the revealed 3.
        board = make_board(revealed=["Board0", "Board1", "Board2"])
        _, number = sm.give_clue(make_ctx(board), sims)
        assert number == MAX_CLUE_NUMBER  # still 4: 6 strong own words remain, all far from any distractor

    def test_revealed_distractor_cannot_affect_the_score(self, tmp_path):
        # An opponent word close enough to hurt the score would normally
        # pull k down -- but if that word is already revealed it must be
        # excluded from the computation entirely, so the clue scores as
        # if it never existed.
        own = [2.0, 1.5] + [_FAR_OWN] * 7
        opponent = [1.4] + [_FAR_NON_OWN] * 7  # right next to the weaker own word
        sm, sims = build(tmp_path, {"clue": make_row(own=own, opponent=opponent)})

        board_with_distractor = make_board()
        _, number_with = sm.give_clue(make_ctx(board_with_distractor), sims)

        board_without_distractor = make_board(revealed=["Board9"])  # the opponent word
        _, number_without = sm.give_clue(make_ctx(board_without_distractor), sims)

        assert number_without >= number_with


class TestNoOwnWordsGuard:
    def test_never_raises_with_no_own_words_left_unrevealed(self, tmp_path):
        sm, sims = build(tmp_path, {"clue": make_row()})
        board = make_board(revealed=[f"Board{i}" for i in range(9)])  # every own word gone
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue in sims.clue_words
        assert number == 1

    def test_picks_the_most_common_legal_clue_when_no_own_words_remain(self, tmp_path):
        rarity = {"common": 1.0, "rare": 99.0}
        rows = {"common": make_row(), "rare": make_row()}
        sm, sims = build(tmp_path, rows, rarity=rarity)
        board = make_board(revealed=[f"Board{i}" for i in range(9)])
        clue, number = sm.give_clue(make_ctx(board), sims)
        assert clue == "common"
        assert number == 1


class TestDeterminism:
    def test_same_board_twice_gives_identical_clue(self, tmp_path):
        rows = {
            "cluea": make_row(own=[2.0, 1.8] + [-20.0] * 7),
            "clueb": make_row(own=[2.2, 1.6] + [-20.0] * 7),
            "cluec": make_row(own=[1.9, 1.7] + [-20.0] * 7),
        }
        sm, sims = build(tmp_path, rows)
        board = make_board()
        a = sm.give_clue(make_ctx(board), sims)
        b = ExpectedWordsSpymaster(space=SPACE, clue_stats=sm.clue_stats).give_clue(make_ctx(make_board()), sims)
        assert a == b


class TestReturnedClueIsAlwaysLegal:
    def test_an_illegal_top_candidate_is_skipped_for_a_legal_runner_up(self, tmp_path):
        # "board0" normalizes to the same string as board word "Board0"
        # -- illegal by codenames.board.is_legal_clue -- and is given a
        # strictly better score (higher own z's, same shape) than the
        # legal alternative, so picking it would be a bug, not a
        # coincidence of tie-breaking.
        rows = {
            "board0": make_row(own=[5.0, 5.0] + [-20.0] * 7),
            "legalclue": make_row(own=[2.0, 2.0] + [-20.0] * 7),
        }
        sm, sims = build(tmp_path, rows)
        board = make_board()
        clue, _ = sm.give_clue(make_ctx(board), sims)
        assert clue == "legalclue"


class TestRarityFiltering:
    def test_a_clue_over_max_rarity_loses_to_a_worse_but_common_one(self, tmp_path):
        rows = {
            "rarebutgreat": make_row(own=[5.0, 5.0] + [-20.0] * 7),
            "commonbutokay": make_row(own=[2.0, 1.5] + [-20.0] * 7),
        }
        rarity = {"rarebutgreat": 99.0, "commonbutokay": 5.0}
        sm, sims = build(tmp_path, rows, rarity=rarity, max_rarity=10.0)
        board = make_board()
        clue, _ = sm.give_clue(make_ctx(board), sims)
        assert clue == "commonbutokay"


class TestScoreBatchProtocol:
    def test_score_batch_matches_top_clues_for_a_single_context(self, tmp_path):
        rows = {
            "cluea": make_row(own=[2.0, 1.8] + [-20.0] * 7),
            "clueb": make_row(own=[2.2, 1.6] + [-20.0] * 7),
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


class TestEveryCandidateGetsAFiniteScore:
    """The whole point of dropping thresholds: there is no 'no valid
    clue' state, so every candidate (and hence every clue) always scores
    finite -- nothing to fall back out of."""

    def test_even_a_clue_surrounded_by_close_distractors_scores_finite(self, tmp_path):
        own = [0.1] * 9
        opponent = [0.1] * 8
        neutral = [0.1] * 7
        assassin = [0.1]
        sm, sims = build(tmp_path, {"clue": make_row(own=own, opponent=opponent, neutral=neutral, assassin=assassin)})
        best_n, scores = sm.score_batch(sims, [make_ctx(make_board())])[0]
        idx = sims.clue_index["clue"]
        assert np.isfinite(scores[idx])
        assert best_n[idx] >= 1
