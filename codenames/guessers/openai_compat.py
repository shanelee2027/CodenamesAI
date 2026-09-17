"""Guesser backed by any OpenAI-compatible chat-completions endpoint.

Deliberately NOT an Anthropic client -- `codenames/guessers/llm.py` is that,
and the two stay separate. One class covers DeepInfra, Groq, Together,
OpenAI itself, and a local vLLM/Ollama server, because they all speak the
same wire format; only `base_url` and `model` change.

**Why this exists.** Evaluating a spymaster means playing games, and games
mean one API call per turn. At Sonnet's rate a sweep fine enough to separate
neighbouring sigma values costs four figures (see docs/log.md), which is why
sigma has only ever been chosen from a cheap per-turn proxy that turned out
to rank sigma=1.5 and sigma=2.5 backwards. An open-weight model at ~1/150th
the price makes the honest experiment affordable.

**The prompt and the parser are imported from llm.py, not re-written.** A
model comparison is only about the model if everything else is byte-identical
-- a reworded prompt would confound it. Cache keys are namespaced by model id
the same way, so Anthropic and OpenAI-compatible rankings share
cache/llm_store.db without colliding.

Reasoning models are a trap for this task, in two ways.

First, cost. Ranking ~17 words against one clue is a single associative
judgment; reasoning tokens bill at the output rate and output is ~90% of the
bill. The repo measured this on Sonnet: 294 output tokens with thinking on
against 108 with it off, no quality difference. Measured head to head on one
25-word position (2026-09-17): Sonnet 5 at effort=medium costs $0.00477 a
call (250 in, 427 out); gpt-oss-120b at `reasoning_effort="low"` costs
$0.000102 (555 completion tokens), 47x cheaper; the same model at `"medium"`
costs $0.000775 (4509 completion tokens), only 6x cheaper. Per *token* the
open-weight model is ~55x cheaper, so most of that advantage is spent on
reasoning tokens -- which is exactly why the effort setting, not the token
price, is what decides whether this is worth doing.

Second, and worse, silence. **`reasoning_effort` is not optional on a model
that reasons by default.** gpt-oss-120b returns its chain of thought in
`reasoning_content` and the answer in `content`; left unset it defaults to
medium, blows through a small `max_tokens` on reasoning alone, and returns
`finish_reason="length"` with `content=""`. `LLMGuesser._parse_ranking` is
deliberately tolerant -- it backfills anything the model omitted from board
order -- so an empty response becomes a complete, well-formed, entirely
fabricated ranking. The first five sampled positions were all this, and
nothing in the output said so.

Board order is role-shuffled (codenames/board.py), so in play that fabricated
ranking is a random guesser rather than a biased one -- which is why it would
have shown up as "the open-weight model is bad" and not as a crash. A sweep
run that way would have been meaningless and would have looked fine.

Hence the `_reject` check below: this guesser raises where the Anthropic one
backfills. The tolerant parser stays correct for its own case (Claude with
thinking disabled either answers or errors) and is left alone.
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path

from codenames.guessers.base import Guesser
from codenames.guessers.llm import _PROMPT_TEMPLATE, LLMGuesser
from codenames.llm_store import LLMResponseCache
from codenames.similarity import SimilarityTensor

# Known OpenAI-compatible hosts. `base_url` may also be passed directly --
# these are shorthands so a config file names a provider, not a URL.
PROVIDERS = {
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "local": ("http://localhost:8000/v1", "LOCAL_API_KEY"),
}


class OpenAICompatGuesser(Guesser):
    def __init__(
        self,
        model: str,
        provider: str = "deepinfra",
        base_url: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str | None = "low",
        min_coverage: float = 0.8,
        cache_path: str | Path | None = None,
        client=None,
    ):
        if provider not in PROVIDERS and base_url is None:
            raise ValueError(f"unknown provider {provider!r}; pass base_url, or one of {list(PROVIDERS)}")
        default_url, default_env = PROVIDERS.get(provider, (None, None))
        self.model = model
        self.provider = provider
        self.base_url = base_url or default_url
        self.api_key_env = api_key_env or default_env
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.min_coverage = min_coverage
        # Count of accepted responses where the model ranked only part of the
        # board. Not zero in practice, and worth reporting rather than hiding:
        # the tail of such a ranking is board order, so these entries must not
        # be reused as if they were a full ranking.
        self.partial_responses = 0
        # Lazy, same reason as LLMGuesser: building a pool that contains this
        # guesser must not require a key unless it is actually used.
        self._client = client
        self._cache: dict[tuple[str, tuple[str, ...], int | None], list[str]] = {}
        self._disk_cache = LLMResponseCache(Path(cache_path)) if cache_path is not None else None
        # Guards the shared dict and the lazy client only -- never held across
        # the network call, so many games' calls overlap. Same contract as
        # LLMGuesser; the arenas rely on it.
        self._lock = threading.Lock()

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                from codenames.env import load_env

                # Key lives in the gitignored .env, never the shell profile --
                # see codenames/env.py for why that matters here specifically.
                load_env()
                key = os.environ.get(self.api_key_env)
                if not key:
                    raise RuntimeError(
                        f"{self.api_key_env} is not set. Put it in the project's .env "
                        f"(see .env.example; the file is gitignored), or export it for "
                        f"one command: `{self.api_key_env}=... python ...`"
                    )
                # max_retries above the default: sweeps deliberately run far
                # more workers than cores because these calls are network-bound,
                # so 429s are expected rather than exceptional.
                self._client = OpenAI(api_key=key, base_url=self.base_url, max_retries=8)
            return self._client

    @property
    def cache_model_id(self) -> str:
        """Namespaced by provider so the same open-weight model served by two
        hosts cannot silently share rankings -- they are different deployments
        and can differ in quantisation, sampling defaults, and version."""
        base = f"{self.provider}/{self.model}"
        return base if self.reasoning_effort is None else f"{base}+effort={self.reasoning_effort}"

    def _query(self, clue: str, candidate_words: list[str], number: int | None) -> list[str]:
        """Raises rather than returning a fabricated ranking. See the module
        docstring: the tolerant parser cannot distinguish "the model ranked
        the board in exactly this order" from "the model returned nothing",
        and on a reasoning model the second is the common case.

        One retry at double the token budget, because the only failure seen
        in practice is truncation and doubling fixes it; a second failure is
        a real problem (wrong model id, a host that ignores
        `reasoning_effort`, a prompt the model won't answer) and should stop
        the run while it is still cheap to stop."""
        count_note = f" for {number} word(s)" if number else ""
        prompt = _PROMPT_TEMPLATE.format(clue=clue, count_note=count_note, words="\n".join(candidate_words))
        budget = self.max_tokens
        problems = []
        for attempt in (1, 2):
            request = {
                "model": self.model,
                "max_completion_tokens": budget,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.reasoning_effort is not None:
                request["reasoning_effort"] = self.reasoning_effort
            choice = self.client.chat.completions.create(**request).choices[0]
            text = choice.message.content or ""
            named = self._named_by_model(text, candidate_words)
            problem = self._reject(choice, text, named, candidate_words, number)
            if problem is None:
                if len(named) < len(candidate_words):
                    self.partial_responses += 1
                return self._ranking_from(text, candidate_words)
            problems.append(f"attempt {attempt} (budget {budget}): {problem}")
            budget *= 2
        raise RuntimeError(
            f"{self.cache_model_id} did not return a usable ranking for clue {clue!r} "
            f"over {len(candidate_words)} words. " + "; ".join(problems) + ". "
            "Not falling back to board order -- that would silently enter a "
            "fabricated ranking into cache/llm_store.db (see this module's docstring)."
        )

    def _reject(
        self, choice, text: str, named: set[str], candidate_words: list[str], number: int | None
    ) -> str | None:
        """None if the response is usable, else why it isn't.

        The test is NOT "did the model rank everything". `_parse_ranking`
        returns the model's own words first and board-order backfill after,
        and a turn only ever consumes the top `number` entries
        (codenames/game.py). So a response is usable exactly when the model
        named enough words to cover what will be read off the top -- a
        16-of-25 ranking is a fine answer to a clue for 2, while a 0-of-25
        one is never an answer to anything.

        Requiring full coverage instead would also bias the sample: the
        positions a model half-answers are the awkward ones, and dropping
        them quietly restricts the measurement to boards it found easy.

        `number + 2` rather than `number` leaves room for a bonus guess
        (Guesser.bonus_guesses) and keeps a margin so the consumed prefix is
        never the very last thing the model said. When `number` is None the
        caller wants a full ranking (score_candidates), so `min_coverage`
        applies instead."""
        n, total = len(named), len(candidate_words)
        if total == 0:
            return None
        if number is not None:
            required = min(total, max(number + 2, 3))
            what = f"a clue for {number}"
        else:
            required = math.ceil(self.min_coverage * total)
            what = f"a full ranking ({self.min_coverage:.0%} coverage)"
        if n >= required:
            return None
        detail = ""
        if choice.finish_reason == "length":
            detail = " -- hit max_completion_tokens, so the reasoning budget ran out before the answer"
        elif not text.strip():
            detail = f" -- empty content (finish_reason={choice.finish_reason})"
        return (
            f"model named only {n}/{total} candidate words, need {required} for {what}; "
            f"the rest would be board-order backfill{detail}"
        )

    @staticmethod
    def _ranking_from(text: str, candidate_words: list[str]) -> list[str]:
        """`LLMGuesser._parse_ranking` with duplicates removed. It filters by
        membership without deduping, so a model that lists a word twice
        yields a ranking longer than the board with a repeat near the top --
        which would make the turn try to reveal the same card twice. Not
        fixed in the shared parser because that would change what every
        cached Anthropic ranking means; done here, where the reasoning
        models that actually do this live."""
        seen: set[str] = set()
        out: list[str] = []
        for w in LLMGuesser._parse_ranking(text, candidate_words):
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    @staticmethod
    def _named_by_model(text: str, candidate_words: list[str]) -> set[str]:
        """The candidate words the response actually contained, recovered the
        same way `_parse_ranking` recovers them so the two cannot disagree."""
        import json
        import re

        match = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            raw = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            raw = []
        allowed = set(candidate_words)
        return {w for w in raw if isinstance(w, str) and w in allowed}

    def _ranked(self, clue: str, candidate_words: list[str], number: int | None) -> list[str]:
        key = (clue, tuple(candidate_words), number)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        cached = self._disk_cache.get(self.cache_model_id, clue, key[1], number) if self._disk_cache else None
        if cached is None:
            cached = self._query(clue, candidate_words, number)
            if self._disk_cache is not None:
                self._disk_cache.put(self.cache_model_id, clue, key[1], number, cached)
        with self._lock:
            self._cache[key] = cached
        return cached

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        ranking = self._ranked(clue, candidate_words, number=None)
        n = len(ranking)
        return {w: float(n - i) for i, w in enumerate(ranking)}

    def rank_candidates(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int | None = None,
        history: list[tuple[str, int]] | None = None,
    ) -> list[str]:
        return self._ranked(clue, candidate_words, number)

    def __repr__(self) -> str:
        return f"OpenAICompatGuesser(provider={self.provider!r}, model={self.model!r})"
