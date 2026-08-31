from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from codenames.board import Role
from codenames.game import TurnResult
from codenames.guessers.base import Guesser
from codenames.guessers.blend import BlendGuesser
from codenames.guessers.confidence_threshold import ConfidenceThresholdGuesser
from codenames.guessers.history_aware import HistoryAwareGuesser
from codenames.guessers.llm import LLMGuesser
from codenames.guessers.noisy import NoisyGuesser
from codenames.guessers.rank_based import RankBasedGuesser
from codenames.guessers.registry import DEFAULT_POOL_CONFIG, held_out_pool, load_pool, training_pool
from codenames.guessers.single_space import SingleSpaceGuesser
from codenames.similarity import SimilarityTensor

BOARD_WORDS = ["Apple", "Banana", "Car", "Doghouse"]
SPACES = ["a", "b"]


@pytest.fixture
def sims(tmp_path):
    # Apple: high in both spaces. Banana: high in 'a', missing in 'b'.
    # Car: missing in 'a', moderate in 'b'. Doghouse: low in both.
    tensor = np.array(
        [[
            [0.9, 0.8],   # Apple
            [0.7, np.nan],  # Banana
            [np.nan, 0.5],  # Car
            [0.1, 0.1],   # Doghouse
        ]],
        dtype=np.float16,
    )
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": SPACES, "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestSingleSpaceGuesser:
    def test_ranks_by_raw_similarity(self, sims):
        g = SingleSpaceGuesser(space="a")
        assert g.rank_candidates("clue", BOARD_WORDS, sims) == ["Apple", "Banana", "Doghouse", "Car"]

    def test_missing_vector_scores_negative_infinity(self, sims):
        g = SingleSpaceGuesser(space="a")
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Car"] == float("-inf")

    def test_missing_vector_ranks_last(self, sims):
        g = SingleSpaceGuesser(space="a")
        ranked = g.rank_candidates("clue", BOARD_WORDS, sims)
        assert ranked[-1] == "Car"


class TestBlendGuesser:
    def test_uniform_weights_average_available_spaces(self, sims):
        g = BlendGuesser(weights={"a": 1.0, "b": 1.0})
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Apple"] == pytest.approx(0.85, abs=1e-2)  # mean(0.9, 0.8)

    def test_renormalizes_over_available_spaces_when_one_missing(self, sims):
        g = BlendGuesser(weights={"a": 1.0, "b": 1.0})
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        # Banana only has space 'a' (0.7) -- should use 0.7, not treat
        # the missing space as 0 (which would silently drag the average down)
        assert scores["Banana"] == pytest.approx(0.7, abs=1e-2)
        assert scores["Car"] == pytest.approx(0.5, abs=1e-2)

    def test_missing_in_all_weighted_spaces_is_negative_infinity(self, sims):
        g = BlendGuesser(weights={"a": 1.0})  # only space 'a', which Car lacks
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Car"] == float("-inf")

    def test_weights_change_ranking(self, tmp_path):
        # X and Y are both present in both spaces (unlike Banana/Car above,
        # which are each present in only one -- renormalizing over a
        # single available space makes its weight cancel out entirely, so
        # that pair can never demonstrate a weight-driven ranking change).
        words = ["X", "Y"]
        tensor = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float16)  # X: a=0.9,b=0.1; Y: a=0.1,b=0.9
        np.save(tmp_path / "similarity_tensor.npy", tensor)
        (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
        (tmp_path / "board_vocab.json").write_text(json.dumps(words))
        (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["a", "b"], "shape": list(tensor.shape)}))
        local_sims = SimilarityTensor.load(cache_dir=tmp_path)

        heavy_a = BlendGuesser(weights={"a": 10.0, "b": 1.0})
        heavy_b = BlendGuesser(weights={"a": 1.0, "b": 10.0})
        assert heavy_a.rank_candidates("clue", words, local_sims)[0] == "X"
        assert heavy_b.rank_candidates("clue", words, local_sims)[0] == "Y"


class TestRankBasedGuesser:
    def test_uses_rank_not_raw_score(self, tmp_path):
        # A has a huge outlier score in space 'a' that dominates any raw
        # average, but loses on rank in both spaces to B. Raw-score
        # blending picks A first; rank-based should pick B first instead.
        words = ["A", "B", "C"]
        tensor = np.array([[[100.0, 0.0], [2.0, 10.0], [1.0, 9.0]]], dtype=np.float16)
        np.save(tmp_path / "similarity_tensor.npy", tensor)
        (tmp_path / "clue_vocab.json").write_text(json.dumps(["clue"]))
        (tmp_path / "board_vocab.json").write_text(json.dumps(words))
        (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["a", "b"], "shape": list(tensor.shape)}))
        local_sims = SimilarityTensor.load(cache_dir=tmp_path)

        raw_blend = BlendGuesser(weights={"a": 1.0, "b": 1.0})
        rank_based = RankBasedGuesser(spaces=["a", "b"])

        assert raw_blend.rank_candidates("clue", words, local_sims)[0] == "A"
        assert rank_based.rank_candidates("clue", words, local_sims)[0] == "B"

    def test_missing_vector_excluded_from_rank_average(self, sims):
        g = RankBasedGuesser(spaces=["a", "b"])
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        # Banana: rank 2 in 'a' (Apple=0.9 first, Banana=0.7 second, Doghouse=0.1 third among valid-in-a)
        # only one valid rank contributes for Banana (missing in 'b')
        assert scores["Banana"] == pytest.approx(-2.0, abs=1e-6)

    def test_missing_in_all_spaces_is_negative_infinity(self, sims):
        g = RankBasedGuesser(spaces=[])
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert all(s == float("-inf") for s in scores.values())


class TestNoisyGuesser:
    def test_same_seed_is_reproducible(self, sims):
        base = SingleSpaceGuesser(space="a")
        g1 = NoisyGuesser(base=base, noise_std=0.5, seed=7)
        g2 = NoisyGuesser(base=base, noise_std=0.5, seed=7)
        assert g1.rank_candidates("clue", BOARD_WORDS, sims) == g2.rank_candidates("clue", BOARD_WORDS, sims)

    def test_repeated_calls_on_the_same_instance_agree(self, sims):
        # The actual bug this guards against: codenames/guessers/base.py's
        # backlog mechanism (and HistoryAwareGuesser's baseline cache)
        # re-score the same clue multiple times across a game and assume
        # they always get the same answer -- true of every guesser here
        # except NoisyGuesser before its noise was made a pure function of
        # (seed, clue, word) instead of a draw from a continuously-
        # advancing RNG stream.
        g = NoisyGuesser(base=SingleSpaceGuesser(space="a"), noise_std=0.5, seed=3)
        first = g.score_candidates("clue", BOARD_WORDS, sims)
        second = g.score_candidates("clue", BOARD_WORDS, sims)
        assert first == second

    def test_noise_is_independent_of_noise_std(self, sims):
        # scripts/run_ablation_study.py's noise-level sweep depends on
        # this: the same seed at a different noise_std should be the same
        # underlying standard-normal draw per (clue, word), just scaled --
        # not an unrelated draw -- so different noise_std levels are
        # directly comparable, not just similarly distributed.
        base = SingleSpaceGuesser(space="a")
        low = NoisyGuesser(base=base, noise_std=1.0, seed=9)
        high = NoisyGuesser(base=base, noise_std=2.0, seed=9)
        base_scores = base.score_candidates("clue", BOARD_WORDS, sims)
        low_delta = low._noise("clue", "Apple")
        high_delta = high._noise("clue", "Apple")
        assert high_delta == pytest.approx(2.0 * low_delta)
        assert base_scores  # sanity: fixture actually has finite scores to perturb

    def test_negative_infinity_is_not_perturbed(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = NoisyGuesser(base=base, noise_std=5.0, seed=1)
        scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert scores["Car"] == float("-inf")

    def test_noise_actually_changes_scores(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = NoisyGuesser(base=base, noise_std=1.0, seed=1)
        base_scores = base.score_candidates("clue", BOARD_WORDS, sims)
        noisy_scores = g.score_candidates("clue", BOARD_WORDS, sims)
        assert noisy_scores["Apple"] != base_scores["Apple"]


class TestConfidenceThresholdGuesser:
    def test_truncates_below_threshold(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = ConfidenceThresholdGuesser(base=base, threshold=0.5)
        # 'a' scores: Apple=0.9, Banana=0.7, Doghouse=0.1, Car=-inf
        ranked = g.rank_candidates("clue", BOARD_WORDS, sims)
        assert ranked == ["Apple", "Banana"]

    def test_threshold_above_everything_returns_empty(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = ConfidenceThresholdGuesser(base=base, threshold=999.0)
        assert g.rank_candidates("clue", BOARD_WORDS, sims) == []

    def test_threshold_below_everything_returns_all_valid(self, sims):
        base = SingleSpaceGuesser(space="a")
        g = ConfidenceThresholdGuesser(base=base, threshold=-1.0)
        ranked = g.rank_candidates("clue", BOARD_WORDS, sims)
        assert ranked == ["Apple", "Banana", "Doghouse"]  # Car still excluded (-inf < -1.0)


class TestGuesserIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Guesser()


class TestRegistry:
    def test_default_pool_config_loads(self):
        entries = load_pool(DEFAULT_POOL_CONFIG)
        assert len(entries) == 3

    def test_default_pool_has_no_held_out_guessers(self):
        # First-pass revision (see docs/log.md): generalization is checked
        # via held-out board words instead of held-out guessers -- see
        # codenames/board.py's load_holdout_wordlist().
        entries = load_pool(DEFAULT_POOL_CONFIG)
        assert all(not e.held_out for e in entries.values())

    def test_default_pool_wraps_each_space_with_noise(self):
        entries = load_pool(DEFAULT_POOL_CONFIG)
        expected = {
            "noisy_glove": "glove",
            "noisy_numberbatch": "numberbatch",
            "noisy_wikipedia2vec": "wikipedia2vec",
        }
        for name, space in expected.items():
            guesser = entries[name].guesser
            assert isinstance(guesser, NoisyGuesser)
            assert isinstance(guesser.base, SingleSpaceGuesser)
            assert guesser.base.space == space

    def test_blend_pool_config_loads(self):
        # configs/guesser_pool_blend.json: a single guesser, a noisy
        # weighted blend across all three spaces (glove/numberbatch/
        # wikipedia2vec) -- exploratory, not the standard training pool.
        from pathlib import Path

        path = Path(__file__).parent.parent / "configs" / "guesser_pool_blend.json"
        entries = load_pool(path)
        assert list(entries) == ["blend"]
        guesser = entries["blend"].guesser
        assert isinstance(guesser, NoisyGuesser)
        assert guesser.noise_std == 0.08
        assert isinstance(guesser.base, BlendGuesser)
        assert guesser.base.weights == {"glove": 0.3, "numberbatch": 0.5, "wikipedia2vec": 0.2}

    def test_accepts_an_already_parsed_config_dict_not_just_a_path(self):
        # scripts/web_inspector.py builds one in-memory pool per noise
        # level by copying and editing the default config dict, so
        # load_pool needs to accept that directly rather than requiring a
        # round-trip through a temp file.
        config = {"guessers": [{"name": "a", "type": "single_space", "params": {"space": "x"}}]}
        entries = load_pool(config)
        assert list(entries) == ["a"]
        assert isinstance(entries["a"].guesser, SingleSpaceGuesser)

    def test_held_out_flag_defaults_to_false(self, tmp_path):
        config = {"guessers": [{"name": "a", "type": "single_space", "params": {"space": "x"}}]}
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(config))
        assert load_pool(path)["a"].held_out is False

    def test_held_out_flag_when_true_is_respected(self, tmp_path):
        config = {
            "guessers": [
                {"name": "a", "type": "single_space", "params": {"space": "x"}, "held_out": True},
                {"name": "b", "type": "single_space", "params": {"space": "y"}},
            ]
        }
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(config))
        assert "a" not in training_pool(path)
        assert "b" in training_pool(path)
        assert "a" in held_out_pool(path)
        assert "b" not in held_out_pool(path)

    def test_inline_anonymous_base_is_not_a_separate_pool_entry(self, tmp_path):
        config = {
            "guessers": [
                {
                    "name": "wrapped",
                    "type": "noisy",
                    "params": {"base": {"type": "single_space", "params": {"space": "x"}}, "noise_std": 0.1, "seed": 1},
                }
            ]
        }
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(config))
        entries = load_pool(path)
        assert list(entries) == ["wrapped"]
        assert isinstance(entries["wrapped"].guesser, NoisyGuesser)
        assert isinstance(entries["wrapped"].guesser.base, SingleSpaceGuesser)
        assert entries["wrapped"].guesser.base.space == "x"

    def test_inline_base_with_invalid_type_raises(self, tmp_path):
        config = {"guessers": [{"name": "bad", "type": "noisy", "params": {"base": 123, "noise_std": 0.1}}]}
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="base"):
            load_pool(path)

    def test_unknown_base_reference_raises(self, tmp_path):
        config = {
            "guessers": [
                {"name": "orphan", "type": "confidence_threshold", "params": {"base": "does_not_exist", "threshold": 0.1}},
            ]
        }
        path = tmp_path / "bad_pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="does_not_exist"):
            load_pool(path)

    def test_duplicate_name_raises(self, tmp_path):
        config = {
            "guessers": [
                {"name": "dup", "type": "single_space", "params": {"space": "a"}},
                {"name": "dup", "type": "single_space", "params": {"space": "b"}},
            ]
        }
        path = tmp_path / "dup_pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="dup"):
            load_pool(path)

    def test_unknown_type_raises(self, tmp_path):
        config = {"guessers": [{"name": "x", "type": "not_a_real_type", "params": {}}]}
        path = tmp_path / "bad_type_pool.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="not_a_real_type"):
            load_pool(path)

    def test_history_aware_pool_config_loads(self):
        from pathlib import Path

        path = Path(__file__).parent.parent / "configs" / "guesser_pool_history_aware.json"
        entries = load_pool(path)
        assert list(entries) == ["history_aware_blend"]
        guesser = entries["history_aware_blend"].guesser
        assert isinstance(guesser, HistoryAwareGuesser)
        assert isinstance(guesser.base, NoisyGuesser)
        assert isinstance(guesser.base.base, BlendGuesser)


def make_turn(clue: str, number: int, guesses: list[tuple[str, Role]], ended_reason: str) -> TurnResult:
    return TurnResult(clue=clue, number=number, guesses=guesses, ended_reason=ended_reason)


class TestGuesserUpdateHistory:
    """The generic backlog bookkeeping on the Guesser base class -- usable
    by any guesser, not just HistoryAwareGuesser (see base.py's
    docstring)."""

    def test_a_miss_creates_a_backlog_entry_for_the_shortfall(self, sims):
        g = SingleSpaceGuesser(space="a")
        turn = make_turn("clue", number=3, guesses=[("Apple", Role.OWN), ("Banana", Role.OPPONENT)], ended_reason="opponent")
        # number=3, only 1 correct before the miss -> 2 still owed.
        history = g.update_history([], "clue", 3, turn, candidates_before_turn=BOARD_WORDS, sims=sims)
        assert history == [("clue", 2)]

    def test_a_clean_finish_creates_no_backlog(self, sims):
        g = SingleSpaceGuesser(space="a")
        turn = make_turn("clue", number=1, guesses=[("Apple", Role.OWN)], ended_reason="exhausted_guesses")
        history = g.update_history([], "clue", 1, turn, candidates_before_turn=BOARD_WORDS, sims=sims)
        assert history == []

    def test_assassin_creates_no_backlog(self, sims):
        # The game is over at that point -- nothing left to carry forward.
        g = SingleSpaceGuesser(space="a")
        turn = make_turn("clue", number=2, guesses=[("Doghouse", Role.ASSASSIN)], ended_reason="assassin")
        history = g.update_history([], "clue", 2, turn, candidates_before_turn=BOARD_WORDS, sims=sims)
        assert history == []

    def test_a_wrong_guess_does_not_touch_existing_owed_counts(self, sims):
        # Both turns share the fixture's one registered clue ("clue") --
        # a real game's backlog entries always name a clue that was
        # actually given earlier, so a fresh, unregistered clue string
        # isn't a realistic scenario here (and would raise KeyError, since
        # scoring an unregistered clue isn't something this guesser is
        # ever asked to do in real play).
        g = SingleSpaceGuesser(space="a")
        turn = make_turn("clue", number=1, guesses=[("Banana", Role.OPPONENT)], ended_reason="opponent")
        history = g.update_history([("clue", 2)], "clue", 1, turn, candidates_before_turn=BOARD_WORDS, sims=sims)
        # The pre-existing entry survives unchanged (Banana isn't "clue"'s
        # own top pick -- Apple is -- so no collision credit either); a
        # new entry for this turn's own miss is added too (number=1, 0
        # correct -> 1 owed).
        assert ("clue", 2) in history
        assert history.count(("clue", 2)) == 1
        assert ("clue", 1) in history

    def test_collision_decrements_and_can_retire_a_backlog_entry(self, sims):
        # SingleSpaceGuesser(space="a") ranks Apple highest for "clue" --
        # if a *different* clue's turn happens to correctly guess Apple
        # too, the "clue" backlog entry should be credited, not left
        # thinking Apple is still separately owed. "clue2" is only ever
        # used as this turn's own clue name here (never re-scored, since
        # this turn didn't end in a miss), so it doesn't need to be a
        # real registered clue in `sims`.
        g = SingleSpaceGuesser(space="a")
        turn = make_turn("clue2", number=1, guesses=[("Apple", Role.OWN)], ended_reason="exhausted_guesses")
        history = g.update_history([("clue", 1)], "clue2", 1, turn, candidates_before_turn=BOARD_WORDS, sims=sims)
        # owed was 1, decremented to 0 by the collision -> entry retired.
        assert history == []

    def test_collision_decrements_without_retiring_when_more_is_still_owed(self, sims):
        g = SingleSpaceGuesser(space="a")
        turn = make_turn("clue2", number=1, guesses=[("Apple", Role.OWN)], ended_reason="exhausted_guesses")
        history = g.update_history([("clue", 2)], "clue2", 1, turn, candidates_before_turn=BOARD_WORDS, sims=sims)
        assert history == [("clue", 1)]


def _make_two_clue_sims(tmp_path, board_words: list[str], clue_scores: dict[str, dict[str, float]]) -> SimilarityTensor:
    clue_words = list(clue_scores)
    tensor = np.array([[[clue_scores[c][w]] for w in board_words] for c in clue_words], dtype=np.float16)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(clue_words))
    (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["a"], "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


class TestHistoryAwareGuesser:
    """z-score-normalized cross-clue merge, checked against hand-computed
    z-scores (see docs/log.md's hubness investigation for why raw scores
    aren't compared directly)."""

    WORDS = ["A", "B", "C"]
    # z(fruit) = {A: 1.2247, B: 0.0, C: -1.2247} -- A is fruit's clear top pick.
    FRUIT = {"A": 0.9, "B": 0.5, "C": 0.1}
    # z(kitchen) = {A: -0.162, B: 1.298, C: -1.136} -- B is kitchen's top
    # pick, but its raw scores are all much closer together than fruit's,
    # which raw-score comparison would miss entirely (kitchen's raw
    # values are all higher than fruit's C and even close to fruit's B).
    KITCHEN = {"A": 0.62, "B": 0.65, "C": 0.60}

    def test_no_history_behaves_exactly_like_the_base_guesser(self, tmp_path):
        sims = _make_two_clue_sims(tmp_path, self.WORDS, {"kitchen": self.KITCHEN, "fruit": self.FRUIT})
        base = SingleSpaceGuesser(space="a")
        g = HistoryAwareGuesser(base=base)
        assert g.rank_candidates("kitchen", self.WORDS, sims, number=1, history=None) == base.rank_candidates(
            "kitchen", self.WORDS, sims
        )
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=1, history=None) == 0
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=1, history=[]) == 0

    def test_competitive_backlog_word_earns_the_bonus_and_is_inserted_by_zscore(self, tmp_path):
        sims = _make_two_clue_sims(tmp_path, self.WORDS, {"kitchen": self.KITCHEN, "fruit": self.FRUIT})
        g = HistoryAwareGuesser(base=SingleSpaceGuesser(space="a"))
        history = [("fruit", 1)]

        # Merged order: kitchen's own ranking is B, A, C (raw: .65, .62, .60).
        # A is also fruit's top pick (z=1.2247), which beats kitchen's own
        # z for A (-0.162) and C (-1.136) but not B's (1.298) -- so A
        # should be spliced in right after B.
        ranked = g.rank_candidates("kitchen", self.WORDS, sims, number=1, history=history)
        assert ranked == ["B", "A", "C"]

        # number=1: A lands at index 1, within the top number+1=2 -> earns
        # the bonus.
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=1, history=history) == 1

    def test_backlog_word_outside_reach_does_not_earn_the_bonus(self, tmp_path):
        sims = _make_two_clue_sims(tmp_path, self.WORDS, {"kitchen": self.KITCHEN, "fruit": self.FRUIT})
        g = HistoryAwareGuesser(base=SingleSpaceGuesser(space="a"))
        history = [("fruit", 1)]
        # number=0: A still lands at index 1 in the merged ranking, which
        # is outside the top number+1=1 -- not competitive enough to
        # spend the one bonus on.
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=0, history=history) == 0

    def test_no_valid_backlog_candidate_means_no_bonus(self, tmp_path):
        # "absent" has no vector at all for any candidate (NaN throughout,
        # same sentinel-handling as a real embedding space with no
        # coverage) -- nothing to spend a bonus on even though it's a
        # real, registered clue with a pending backlog entry.
        nan = float("nan")
        sims = _make_two_clue_sims(
            tmp_path, self.WORDS, {"kitchen": self.KITCHEN, "absent": {"A": nan, "B": nan, "C": nan}}
        )
        g = HistoryAwareGuesser(base=SingleSpaceGuesser(space="a"))
        history = [("absent", 1)]
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=1, history=history) == 0
        assert g.rank_candidates("kitchen", self.WORDS, sims, number=1, history=history) == ["B", "A", "C"]

    def test_a_satisfied_backlog_does_not_earn_a_bonus_on_a_later_turn(self, tmp_path):
        """Regression test for a real bug report: a NoisyGuesser-wrapped
        HistoryAwareGuesser correctly used its bonus guess to satisfy a
        backlog entry, but a *later* turn still claimed an unearned bonus
        for the same (already-resolved) backlog. Root cause: NoisyGuesser
        used to draw fresh random noise on every call, so re-scoring
        "fruit" during update_history's retrospective collision check
        could disagree with what was actually guessed during real play --
        fixed by making its noise a pure function of (seed, clue, word).
        This test wraps a NoisyGuesser specifically because a deterministic
        base guesser could never have exposed this."""
        sims = _make_two_clue_sims(tmp_path, self.WORDS, {"kitchen": self.KITCHEN, "fruit": self.FRUIT})
        g = HistoryAwareGuesser(base=NoisyGuesser(base=SingleSpaceGuesser(space="a"), noise_std=0.2, seed=5))

        # Turn 1 (some earlier clue "fruit", number=1): A is fruit's clear
        # top pick, but suppose the guesser actually got it wrong somehow
        # -- simplest way to reach the same state as the bug report
        # (a genuine backlog entry) is to hand it in directly, same as
        # TestGuesserUpdateHistory does.
        history = [("fruit", 1)]

        # Turn 2 ("kitchen", number=1): merged ranking should be B, A, C
        # (same z-score computation as the deterministic test above, since
        # noise is fixed per (clue, word) and both calls below use the
        # exact same clue/candidates). The guesser claims its bonus and
        # correctly guesses both B (kitchen's own pick) and A (fruit's
        # backlog pick).
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=1, history=history) == 1
        ranked = g.rank_candidates("kitchen", self.WORDS, sims, number=1, history=history)
        guesses = [(w, Role.OWN) for w in ranked[:2]]  # number=1 + bonus=1
        turn = make_turn("kitchen", number=1, guesses=guesses, ended_reason="exhausted_guesses")
        candidates_before_turn = self.WORDS
        new_history = g.update_history(history, "kitchen", 1, turn, candidates_before_turn, sims)

        # The backlog was actually satisfied this turn -- it must not
        # survive into turn 3, and turn 3 must not claim an unearned bonus.
        assert new_history == []
        assert g.bonus_guesses("kitchen", self.WORDS, sims, number=1, history=new_history) == 0


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._responses.pop(0))


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.messages = _FakeMessages(responses)


class _SlowFakeMessages:
    """Ignores the input entirely and just sleeps -- for proving many
    calls run concurrently rather than one at a time, not for checking
    ranking content."""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        return _FakeResponse("[]")


class _SlowFakeClient:
    def __init__(self, delay: float):
        self.messages = _SlowFakeMessages(delay)


class TestLLMGuesser:
    WORDS = ["Apple", "Banana", "Car", "Doghouse"]

    def test_rank_candidates_uses_the_models_ranking(self):
        client = _FakeClient(['["Car", "Apple", "Doghouse", "Banana"]'])
        g = LLMGuesser(client=client)
        assert g.rank_candidates("fruit", self.WORDS, sims=None, number=2) == ["Car", "Apple", "Doghouse", "Banana"]

    def test_malformed_response_falls_back_to_original_order(self):
        client = _FakeClient(["not json at all"])
        g = LLMGuesser(client=client)
        assert g.rank_candidates("fruit", self.WORDS, sims=None) == self.WORDS

    def test_partial_response_appends_missing_words_in_original_order(self):
        # Model only mentions two of the four words.
        client = _FakeClient(['["Banana", "Car"]'])
        g = LLMGuesser(client=client)
        ranked = g.rank_candidates("fruit", self.WORDS, sims=None)
        assert ranked == ["Banana", "Car", "Apple", "Doghouse"]

    def test_repeated_calls_with_the_same_inputs_are_cached(self):
        client = _FakeClient(['["Apple", "Banana", "Car", "Doghouse"]'])
        g = LLMGuesser(client=client)
        first = g.rank_candidates("fruit", self.WORDS, sims=None, number=1)
        second = g.rank_candidates("fruit", self.WORDS, sims=None, number=1)
        assert first == second
        assert len(client.messages.calls) == 1

    def test_different_number_is_a_cache_miss(self):
        client = _FakeClient(['["Apple", "Banana", "Car", "Doghouse"]', '["Banana", "Apple", "Car", "Doghouse"]'])
        g = LLMGuesser(client=client)
        g.rank_candidates("fruit", self.WORDS, sims=None, number=1)
        g.rank_candidates("fruit", self.WORDS, sims=None, number=2)
        assert len(client.messages.calls) == 2

    def test_score_candidates_is_monotonic_with_the_ranking(self):
        client = _FakeClient(['["Car", "Apple", "Doghouse", "Banana"]'])
        g = LLMGuesser(client=client)
        scores = g.score_candidates("fruit", self.WORDS, sims=None)
        ranked_by_score = sorted(self.WORDS, key=lambda w: -scores[w])
        assert ranked_by_score == ["Car", "Apple", "Doghouse", "Banana"]

    def test_registry_builds_an_llm_guesser(self, tmp_path):
        import json as json_module

        from codenames.guessers.registry import load_pool

        config = {"guessers": [{"name": "llm", "type": "llm", "params": {}}]}
        path = tmp_path / "pool.json"
        path.write_text(json_module.dumps(config))
        entries = load_pool(path)
        assert isinstance(entries["llm"].guesser, LLMGuesser)

    def test_disk_cache_avoids_a_second_api_call(self, tmp_path):
        db_path = tmp_path / "store.db"
        client = _FakeClient(['["Car", "Apple", "Doghouse", "Banana"]'])
        g = LLMGuesser(client=client, cache_path=db_path)
        g.rank_candidates("fruit", self.WORDS, sims=None, number=2)

        # Fresh instance, same disk cache, no client at all -- must not
        # need one, since the disk cache should answer without a query.
        g2 = LLMGuesser(client=None, cache_path=db_path)
        assert g2.rank_candidates("fruit", self.WORDS, sims=None, number=2) == ["Car", "Apple", "Doghouse", "Banana"]

    def test_disk_cache_survives_process_restart_semantics(self, tmp_path):
        # Simulated by two separate LLMGuesser instances against the same
        # db file, rather than an actual subprocess -- what matters is the
        # cache being on disk, not in either instance's memory.
        db_path = tmp_path / "store.db"
        client = _FakeClient(['["Apple", "Banana", "Car", "Doghouse"]'])
        LLMGuesser(client=client, cache_path=db_path).rank_candidates("fruit", self.WORDS, sims=None, number=1)

        second_client = _FakeClient(['["Doghouse", "Car", "Banana", "Apple"]'])
        g2 = LLMGuesser(client=second_client, cache_path=db_path)
        assert g2.rank_candidates("fruit", self.WORDS, sims=None, number=1) == ["Apple", "Banana", "Car", "Doghouse"]
        assert len(second_client.messages.calls) == 0

    def test_concurrent_calls_overlap_instead_of_serializing(self):
        # codenames/two_team_gpu_arena.py shares one LLMGuesser across many
        # games' threads specifically so their network calls overlap --
        # this proves the locking added for that doesn't accidentally
        # serialize the calls themselves back onto one thread.
        n, delay = 5, 0.2
        client = _SlowFakeClient(delay=delay)
        g = LLMGuesser(client=client)

        start = time.monotonic()
        threads = [
            threading.Thread(target=g.rank_candidates, args=(f"clue{i}", self.WORDS), kwargs={"sims": None, "number": 1})
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start

        assert len(client.messages.calls) == n
        # Serialized, this would take n * delay; concurrent, it should be
        # close to one delay. The midpoint is a generous margin against
        # scheduling noise while still failing if calls are serialized.
        assert elapsed < n * delay / 2
