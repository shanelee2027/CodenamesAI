from __future__ import annotations

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from codenames.guessers.base import Guesser
from codenames.guessers.llm import LLMGuesser
from codenames.guessers.noisy import NoisyGuesser
from codenames.guessers.openai_compat import ATTEMPTS, OpenAICompatGuesser
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


class TestNoisyGuesser:
    def test_same_seed_is_reproducible(self, sims):
        base = SingleSpaceGuesser(space="a")
        g1 = NoisyGuesser(base=base, noise_std=0.5, seed=7)
        g2 = NoisyGuesser(base=base, noise_std=0.5, seed=7)
        assert g1.rank_candidates("clue", BOARD_WORDS, sims) == g2.rank_candidates("clue", BOARD_WORDS, sims)

    def test_repeated_calls_on_the_same_instance_agree(self, sims):
        # Noise is a pure function of (seed, clue, word), not a draw from a
        # continuously-advancing RNG stream, so re-scoring agrees.
        g = NoisyGuesser(base=SingleSpaceGuesser(space="a"), noise_std=0.5, seed=3)
        first = g.score_candidates("clue", BOARD_WORDS, sims)
        second = g.score_candidates("clue", BOARD_WORDS, sims)
        assert first == second

    def test_noise_is_independent_of_noise_std(self, sims):
        # A noise-level sweep depends on this: the same seed at a
        # different noise_std should be the same
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

    def test_accepts_an_already_parsed_config_dict_not_just_a_path(self):
        # A noise sweep builds one in-memory pool per level by copying
        # and editing the default config dict, so load_pool needs to
        # accept that directly rather than requiring a round-trip through
        # a temp file.
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
                {"name": "orphan", "type": "noisy", "params": {"base": "does_not_exist", "noise_std": 0.1}},
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

def _make_two_clue_sims(tmp_path, board_words: list[str], clue_scores: dict[str, dict[str, float]]) -> SimilarityTensor:
    clue_words = list(clue_scores)
    tensor = np.array([[[clue_scores[c][w]] for w in board_words] for c in clue_words], dtype=np.float16)
    np.save(tmp_path / "similarity_tensor.npy", tensor)
    (tmp_path / "clue_vocab.json").write_text(json.dumps(clue_words))
    (tmp_path / "board_vocab.json").write_text(json.dumps(board_words))
    (tmp_path / "similarity_meta.json").write_text(json.dumps({"spaces": ["a"], "shape": list(tensor.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


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


class TestLLMGuesserEffort:
    """`effort` is opt-in per model: Opus 5 takes it, the Haiku 4.5 that
    DEFAULT_MODEL points at rejects it, so it must never be sent
    unconditionally. See LLMGuesser.__init__."""

    WORDS = ["Apple", "Banana", "Car", "Doghouse"]
    RANKING = '["Car", "Apple", "Doghouse", "Banana"]'

    def _rank(self, **kwargs):
        client = _FakeClient([self.RANKING])
        guesser = LLMGuesser(client=client, **kwargs)
        guesser.rank_candidates("vehicle", self.WORDS, None)
        return client.messages.calls[0]

    def test_without_effort_thinking_is_disabled_and_no_output_config(self):
        call = self._rank()
        assert call["thinking"] == {"type": "disabled"}
        assert "output_config" not in call

    def test_with_effort_sends_output_config_and_leaves_thinking_alone(self):
        call = self._rank(model="claude-opus-5", effort="medium")
        assert call["output_config"] == {"effort": "medium"}
        # Thinking must NOT be disabled here: on Opus 5 that is the
        # discouraged way to cut cost (it can leak <thinking> tags into the
        # visible response); lowering effort is the supported way.
        assert "thinking" not in call

    def test_effort_is_part_of_the_cache_identity(self):
        base = LLMGuesser(model="claude-opus-5")
        medium = LLMGuesser(model="claude-opus-5", effort="medium")
        high = LLMGuesser(model="claude-opus-5", effort="high")
        assert base.cache_model_id == "claude-opus-5"
        assert medium.cache_model_id != high.cache_model_id
        assert medium.cache_model_id != base.cache_model_id

    def test_different_efforts_do_not_share_disk_cache_entries(self, tmp_path):
        """Otherwise changing effort would silently keep serving rankings
        produced at the old one -- results that were paid for under
        different conditions than the ones being reported."""
        db = tmp_path / "store.db"
        first = _FakeClient([self.RANKING])
        LLMGuesser(model="claude-opus-5", effort="medium", client=first, cache_path=db).rank_candidates(
            "vehicle", self.WORDS, None
        )
        assert len(first.messages.calls) == 1

        # Same model, same prompt, different effort -> must miss the cache.
        second = _FakeClient(['["Banana", "Car", "Apple", "Doghouse"]'])
        LLMGuesser(model="claude-opus-5", effort="high", client=second, cache_path=db).rank_candidates(
            "vehicle", self.WORDS, None
        )
        assert len(second.messages.calls) == 1

        # ...while the identical configuration still hits it.
        third = _FakeClient([])  # would IndexError if it tried to call
        ranking = LLMGuesser(
            model="claude-opus-5", effort="medium", client=third, cache_path=db
        ).rank_candidates("vehicle", self.WORDS, None)
        assert third.messages.calls == []
        assert ranking[0] == "Car"


class _FakeOpenAIChoice:
    def __init__(self, content: str | None, finish_reason: str = "stop"):
        self.finish_reason = finish_reason
        self.message = SimpleNamespace(content=content)


class _FakeCompletions:
    """Returns each scripted response in turn, so a test can script a first
    truncated call followed by a good one and assert on the retry."""

    def __init__(self, choices: list[_FakeOpenAIChoice]):
        self._choices = list(choices)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        choice = self._choices.pop(0) if self._choices else self._choices_exhausted()
        return SimpleNamespace(choices=[choice])

    @staticmethod
    def _choices_exhausted():
        raise AssertionError("guesser made more calls than the test scripted")


class _FakeOpenAIClient:
    def __init__(self, choices: list[_FakeOpenAIChoice]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(choices))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


class TestOpenAICompatGuesserStrictness:
    """A reasoning model that overruns its token budget returns
    `finish_reason="length"` with empty content. `LLMGuesser._parse_ranking`
    is deliberately tolerant and turns that into a complete board-order
    ranking -- a fabricated answer that is indistinguishable from a real one
    and that would be written into cache/llm_store.db as if it had been paid
    for. This guesser must raise instead. See codenames/guessers/openai_compat.py.
    """

    WORDS = ["Apple", "Banana", "Car", "Doghouse", "Elephant"]
    RANKING = '["Car", "Apple", "Doghouse", "Banana", "Elephant"]'

    def _guesser(self, choices, **kwargs):
        client = _FakeOpenAIClient(choices)
        return OpenAICompatGuesser(model="openai/gpt-oss-120b", client=client, **kwargs), client

    def test_good_response_is_returned(self):
        g, _ = self._guesser([_FakeOpenAIChoice(self.RANKING)])
        assert g.rank_candidates("vehicle", self.WORDS, None, number=2) == [
            "Car", "Apple", "Doghouse", "Banana", "Elephant"
        ]

    def test_truncated_response_raises_rather_than_returning_board_order(self):
        g, _ = self._guesser([_FakeOpenAIChoice("", "length")] * ATTEMPTS)
        with pytest.raises(RuntimeError, match="max_completion_tokens"):
            g.rank_candidates("vehicle", self.WORDS, None, number=2)

    def test_truncation_retries_once_at_double_the_budget(self):
        g, client = self._guesser(
            [_FakeOpenAIChoice("", "length"), _FakeOpenAIChoice(self.RANKING)],
            max_tokens=1000,
        )
        assert g.rank_candidates("vehicle", self.WORDS, None, number=2)[0] == "Car"
        assert [c["max_completion_tokens"] for c in client.calls] == [1000, 2000]

    def test_response_naming_too_few_words_is_rejected(self):
        """The parser backfills, so length proves nothing -- what matters is
        whether the model named enough to cover what the turn consumes."""
        partial = _FakeOpenAIChoice('["Car", "Apple"]')
        g, _ = self._guesser([partial] * ATTEMPTS)
        with pytest.raises(RuntimeError, match="named only 2/5"):
            g.rank_candidates("vehicle", self.WORDS, None, number=3)

    def test_partial_ranking_is_accepted_when_it_covers_the_guesses(self):
        """A 16-of-25 ranking is a fine answer to a clue for 2: the turn only
        reads the top `number` entries, and those are all model-ranked.
        Rejecting it would also drop exactly the awkward positions and bias
        the sample toward boards the model found easy."""
        g, _ = self._guesser([_FakeOpenAIChoice('["Car", "Apple", "Doghouse"]')])
        assert g.rank_candidates("vehicle", self.WORDS, None, number=1)[:3] == [
            "Car", "Apple", "Doghouse"
        ]
        assert g.partial_responses == 1

    def test_full_ranking_request_still_needs_coverage(self):
        """score_candidates passes number=None and wants the whole board
        ordered, so the prefix rule does not apply there."""
        partial = _FakeOpenAIChoice('["Car", "Apple"]')
        g, _ = self._guesser([partial] * ATTEMPTS)
        with pytest.raises(RuntimeError, match="full ranking"):
            g.score_candidates("vehicle", self.WORDS, None)

    def test_a_degenerate_response_is_retried_past_the_second_attempt(self):
        """The real failure: well-formed JSON filled with one repeated token
        (["gross","gross",...]) rather than truncation. It is transient at
        roughly 1 call in 4, and a board needs ~30 calls, so stopping at two
        attempts discarded ~20% of boards in an 11-setting sweep."""
        degenerate = _FakeOpenAIChoice('["vehicle", "vehicle", "vehicle"]')
        g, client = self._guesser([degenerate, degenerate, _FakeOpenAIChoice(self.RANKING)])
        assert g.rank_candidates("vehicle", self.WORDS, None, number=2)[0] == "Car"
        assert len(client.calls) == 3, "must not give up after two attempts"

    def test_frequency_penalty_is_sent_only_on_retries(self):
        """A repetition loop is what frequency_penalty exists to break, but
        the first call must stay on the provider's defaults so the common
        path -- and everything already cached -- is unchanged."""
        g, client = self._guesser(
            [_FakeOpenAIChoice('["vehicle", "vehicle"]'), _FakeOpenAIChoice(self.RANKING)])
        g.rank_candidates("vehicle", self.WORDS, None, number=2)
        assert "frequency_penalty" not in client.calls[0]
        assert client.calls[1]["frequency_penalty"] > 0

    def test_budget_doubles_once_and_then_stops_growing(self):
        """Doubling guards the truncation case, but growing it every attempt
        just buys a longer degenerate response."""
        bad = _FakeOpenAIChoice('["vehicle"]')
        g, client = self._guesser([bad] * ATTEMPTS, max_tokens=1000)
        with pytest.raises(RuntimeError):
            g.rank_candidates("vehicle", self.WORDS, None, number=2)
        budgets = [c["max_completion_tokens"] for c in client.calls]
        assert budgets == [1000] + [2000] * (ATTEMPTS - 1)

    def test_duplicate_words_are_removed(self):
        """A repeated word would otherwise survive into the ranking and make
        the turn try to reveal the same card twice."""
        g, _ = self._guesser([_FakeOpenAIChoice('["Car", "Car", "Apple", "Doghouse", "Banana"]')])
        ranking = g.rank_candidates("vehicle", self.WORDS, None, number=2)
        assert ranking[:4] == ["Car", "Apple", "Doghouse", "Banana"]
        assert len(ranking) == len(set(ranking)) == len(self.WORDS)

    def test_a_rejected_response_is_never_cached(self, tmp_path):
        """The whole point: a fabricated ranking in the store would outlive
        the run that produced it and be served to every later comparison."""
        db = tmp_path / "store.db"
        g, _ = self._guesser([_FakeOpenAIChoice("", "length")] * ATTEMPTS, cache_path=db)
        with pytest.raises(RuntimeError):
            g.rank_candidates("vehicle", self.WORDS, None, number=2)
        assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0

    def test_reasoning_effort_defaults_to_low_and_is_sent(self):
        """Unset means the provider's default, which on gpt-oss-120b is
        medium -- 8x the tokens for the same one-shot judgment."""
        g, client = self._guesser([_FakeOpenAIChoice(self.RANKING)])
        assert g.reasoning_effort == "low"
        g.rank_candidates("vehicle", self.WORDS, None, number=2)
        assert client.calls[0]["reasoning_effort"] == "low"

    def test_effort_is_part_of_the_cache_identity(self):
        low = OpenAICompatGuesser(model="m", client=object())
        high = OpenAICompatGuesser(model="m", reasoning_effort="high", client=object())
        assert low.cache_model_id != high.cache_model_id
        assert low.cache_model_id == "deepinfra/m+effort=low"


class TestBuildGuesser:
    """One-line guesser specs must land on the cache identities already in
    cache/llm_store.db, or every past game would silently re-bill."""

    def test_anthropic_spec_keeps_the_stored_cache_identity(self):
        from codenames.guessers.registry import build_guesser
        assert build_guesser("anthropic:claude-sonnet-5:medium").cache_model_id == "claude-sonnet-5+effort=medium"
        assert build_guesser("anthropic:claude-sonnet-5").cache_model_id == "claude-sonnet-5"

    def test_openai_compatible_spec_defaults_to_low_effort(self):
        from codenames.guessers.registry import build_guesser
        g = build_guesser("deepinfra:openai/gpt-oss-120b")
        assert g.cache_model_id == "deepinfra/openai/gpt-oss-120b+effort=low"

    def test_a_bare_name_comes_from_the_pool_config(self):
        from codenames.guessers.registry import build_guesser
        assert isinstance(build_guesser("noisy_glove"), NoisyGuesser)

    @pytest.mark.parametrize("spec", ["nowhere:some-model", "anthropic:", "not_in_the_pool"])
    def test_bad_specs_raise(self, spec):
        from codenames.guessers.registry import build_guesser
        with pytest.raises((ValueError, KeyError)):
            build_guesser(spec)
