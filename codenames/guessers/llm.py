"""An external, independent listener -- for evaluation, never training
(see docs/log.md). Every other guesser in the pool was handcrafted
specifically to be a training target: the codemaster learns to please
exactly the listeners it's shown. That makes a high self-play score
ambiguous -- it could mean "this codemaster is genuinely good," or it
could just mean "this codemaster and this guesser happen to share the
same blind spots." A real LLM was never part of that training loop, so
scoring against it breaks the coupling: it's the closest cheap proxy
this project has for "would an actual human guess this."

Deliberately NOT used in scripts/generate_training_data.py or anywhere
in the training path -- that samples millions of (board, clue, guesser)
triples, and a real API call per example would be far too slow and
expensive. This is wired into evaluation the same way any other guesser
is (codenames/two_team_arena.py, scripts/run_two_team_arena.py), just
never into training_pool()'s sampling.

One call per (clue, candidate_words, number) triple, not one call per
candidate word -- the model ranks the whole remaining board at once, the
same shape of question a real guesser answers each turn. Responses are
cached per exact (clue, candidates, number) key: this keeps repeated
calls for the same inputs both cheap (no redundant spend) and consistent
with every other guesser's "re-scoring the same input reproduces the
same answer" property, which codenames/guessers/base.py's backlog
mechanism depends on (see codenames/guessers/noisy.py's docstring for
why that property matters) -- real LLM sampling isn't perfectly
deterministic even at temperature 0, so without this cache reusing an
LLMGuesser inside HistoryAwareGuesser would reintroduce that exact bug.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from codenames.guessers.base import Guesser
from codenames.llm_store import LLMResponseCache
from codenames.similarity import SimilarityTensor

DEFAULT_MODEL = "claude-haiku-4-5"

_PROMPT_TEMPLATE = """You are playing the guesser role in the board game Codenames. Your \
spymaster gave the clue "{clue}"{count_note}. Here are the words still available to guess:

{words}

Rank ALL of the words above by how likely you would guess them for this clue, most \
likely first. Respond with ONLY a JSON array of the words as strings, e.g. \
["word1", "word2", ...] -- no other text. Every word listed above must appear exactly \
once in your answer."""


class LLMGuesser(Guesser):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
        client=None,
        cache_path: str | Path | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Constructed lazily (not at __init__ time) so importing/building a
        # guesser pool that includes this one doesn't require an API key
        # unless it's actually used -- e.g. tests that inject a fake
        # `client` never touch the real SDK at all.
        self._client = client
        self._cache: dict[tuple[str, tuple[str, ...], int | None], list[str]] = {}
        # Disk-backed, so a query already paid for in a past run (or by a
        # sibling worker process in this one) is never re-billed -- see
        # codenames/llm_store.py. Opt-in via cache_path: off by default so
        # tests with a fake client never touch the filesystem.
        self._disk_cache = LLMResponseCache(Path(cache_path)) if cache_path is not None else None
        # One LLMGuesser instance is shared across every game in a batch
        # (codenames/two_team_gpu_arena.py plays them concurrently on a
        # thread pool specifically so their network calls overlap -- see
        # that module's docstring), so `_cache`/`_client` construction need
        # to be safe under concurrent access. This guards only the shared
        # dict/lazy-init, never the network call itself -- holding it
        # across `_query` would serialize every call back onto one thread
        # and defeat the whole point of running them concurrently.
        self._lock = threading.Lock()

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                import os

                import anthropic

                # Some API keys are "identity-linked" (Console access tied
                # to an org/SSO identity rather than a plain personal
                # account) and are rejected with a 400 unless every request
                # names which workspace it acts in. This is per-account
                # setup, not something the library should hardcode --
                # ANTHROPIC_WORKSPACE_ID is unset (and this header omitted)
                # for a plain personal key.
                workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
                headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
                self._client = anthropic.Anthropic(default_headers=headers)
            return self._client

    def _ranked(self, clue: str, candidate_words: list[str], number: int | None) -> list[str]:
        key = (clue, tuple(candidate_words), number)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Neither lock is held past this point while the disk read/API
        # call happens -- both are what let many games' calls actually run
        # concurrently instead of one at a time (see __init__'s note).
        cached = self._disk_cache.get(self.model, clue, key[1], number) if self._disk_cache else None
        if cached is None:
            cached = self._query(clue, candidate_words, number)
            if self._disk_cache is not None:
                self._disk_cache.put(self.model, clue, key[1], number, cached)

        with self._lock:
            self._cache[key] = cached
        return cached

    def _query(self, clue: str, candidate_words: list[str], number: int | None) -> list[str]:
        count_note = f" for {number} word(s)" if number else ""
        prompt = _PROMPT_TEMPLATE.format(clue=clue, count_note=count_note, words="\n".join(candidate_words))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((block.text for block in response.content if block.type == "text"), "")
        return self._parse_ranking(text, candidate_words)

    @staticmethod
    def _parse_ranking(text: str, candidate_words: list[str]) -> list[str]:
        """Never lets a malformed or partial response drop a word from
        consideration: anything the model didn't mention (or the whole
        response, if it couldn't be parsed at all) falls back to the
        board's own original order, appended after whatever the model did
        rank."""
        match = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            raw = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            raw = []
        valid = [w for w in raw if isinstance(w, str) and w in candidate_words]
        missing = [w for w in candidate_words if w not in valid]
        return valid + missing

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
        return f"LLMGuesser(model={self.model!r})"
