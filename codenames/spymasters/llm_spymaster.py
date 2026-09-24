"""An LLM as the spymaster: shown the board with every card's team, asked for
a clue and a number. A reference opponent, not a model of ours -- it answers
"does the learned listener beat a strong general-purpose model at the same
job?", with the same LLM guessing for both sides.

**It is told this arena's rules, not the board game's.** codenames/game.py
has the guesser take exactly `number` guesses, stopping at the first one that
is not its team's, with no bonus guess and no passing, and caps the number at
MAX_CLUE_NUMBER. The learned listener's reward is built on exactly that rule
(codenames/pl_reward.py), so leaving the LLM to assume the tabletop rules
would handicap it: it might count on a guesser that stops when unsure.

**Legality is the arena's rule too**: `is_legal_clue` against every word on
the board, revealed or not -- the same check every other spymaster's
candidates pass through (codenames/clue_search.py). The clue must be a single
alphabetic word, but need NOT be in our clue vocabulary: the LLM guesser does
not need embeddings, and restricting Sonnet to our pool would make it play
our game rather than its own. An illegal or malformed answer is re-asked with
the reason stated; after ATTEMPTS it raises rather than inventing a clue.

Answers are cached per (model, exact prompt) in llm_store.db's
`spymaster_responses` table, with the raw text and token usage
(codenames/llm_store.py::SpymasterResponseCache).
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from codenames.board import MAX_CLUE_NUMBER, Role, is_legal_clue
from codenames.llm_store import DEFAULT_DB_PATH, SpymasterResponseCache
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import Spymaster, TurnContext

ATTEMPTS = 4

_PROMPT = """You are the spymaster for your team in the board game Codenames. You can see \
which team every card belongs to; your guesser cannot.

Your team's words still to find ({n_own}): {own}
The other team's words still unrevealed ({n_opp}): {opp}
Neutral words still unrevealed: {neutral}
The assassin: {assassin}
Already revealed (off the table): {revealed}

The rules in this game:
- Give a one-word clue and a number from 1 to {max_n}.
- Your guesser then guesses exactly that many words, one at a time, and stops at \
the first guess that is not one of your team's words. There is no extra guess, and \
the guesser cannot pass.
- Guessing a neutral word ends your turn. Guessing the other team's word ends your \
turn and reveals it for them. Guessing the assassin loses the game immediately.
- The first team to find all of its words wins.
- The clue must be a single English word. It must not be any word on the board \
(including revealed ones), contain one, be contained in one, or be a form of one \
(plural, verb form, or shared root).

Respond with ONLY a JSON object, e.g. {{"clue": "ocean", "number": 2}} -- no other text."""

_RETRY = ('\n\nYour previous answer was not allowed: {reason} Give a different clue, again as '
          'ONLY a JSON object.')


def board_prompt(board) -> str:
    """The whole prompt for one board state, from the spymaster's own side
    (an OpponentBoardView already has OWN and OPPONENT swapped)."""
    def words(role: Role) -> list[str]:
        return board.words_by_role(role, unrevealed_only=True)

    own, opp = words(Role.OWN), words(Role.OPPONENT)
    revealed = [w for w in board.words if board.is_revealed(w)]
    return _PROMPT.format(
        n_own=len(own), own=", ".join(own), n_opp=len(opp), opp=", ".join(opp),
        neutral=", ".join(words(Role.NEUTRAL)) or "(none)",
        assassin=", ".join(words(Role.ASSASSIN)) or "(already revealed)",
        revealed=", ".join(revealed) or "(none)", max_n=MAX_CLUE_NUMBER)


def parse_clue(text: str, board_words) -> tuple[str, int] | str:
    """(clue, number), or the reason the answer is unusable."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict) or "clue" not in obj or "number" not in obj:
        return 'it was not a JSON object with "clue" and "number".'
    clue = str(obj["clue"]).strip()
    try:
        number = int(obj["number"])
    except (TypeError, ValueError):
        return f"the number {obj['number']!r} is not an integer."
    if not re.fullmatch(r"[A-Za-z]+", clue):
        return f"'{clue}' is not a single alphabetic word."
    if not 1 <= number <= MAX_CLUE_NUMBER:
        return f"the number must be between 1 and {MAX_CLUE_NUMBER}, not {number}."
    if not is_legal_clue(clue, board_words):
        return f"'{clue}' matches, contains, or shares a root with a word on the board."
    return clue.lower(), number


class LLMSpymaster(Spymaster):
    def __init__(self, model: str = "claude-sonnet-5", effort: str | None = "medium",
                 max_tokens: int = 16000, cache_path: str | Path | None = DEFAULT_DB_PATH,
                 client=None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self._client = client
        self._cache: SpymasterResponseCache | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.model if self.effort is None else f"{self.model}+effort={self.effort}"

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                import anthropic

                from codenames.env import load_env

                # Same key handling as codenames/guessers/llm.py: from the
                # gitignored .env, never the shell profile.
                load_env()
                workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
                headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
                self._client = anthropic.Anthropic(default_headers=headers, max_retries=8)
            return self._client

    @property
    def cache(self) -> SpymasterResponseCache | None:
        # Lazy for the same reason as the client: spawned arena workers build
        # this object from its params, and an open sqlite connection does not
        # survive the trip.
        with self._lock:
            if self._cache is None and self.cache_path is not None:
                self._cache = SpymasterResponseCache(Path(self.cache_path))
            return self._cache

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        """One clue: the LLM is asked for its best, not a ranked list, so k
        is ignored (the interface allows returning fewer). The score is 0.0,
        having no meaning here."""
        prompt = board_prompt(ctx.board)
        cached = self.cache.get(self.model_id, prompt) if self.cache else None
        if cached is not None:
            return [(cached[0], cached[1], 0.0)]
        clue, number = self._ask(prompt, list(ctx.board.words))
        return [(clue, number, 0.0)]

    def _ask(self, prompt: str, board_words: list[str]) -> tuple[str, int]:
        ask, reasons, tin, tout = prompt, [], 0, 0
        for attempt in range(1, ATTEMPTS + 1):
            request = {"model": self.model, "max_tokens": self.max_tokens,
                       "messages": [{"role": "user", "content": ask}]}
            if self.effort is not None:
                request["output_config"] = {"effort": self.effort}
            response = self.client.messages.create(**request)
            tin += response.usage.input_tokens
            tout += response.usage.output_tokens
            text = next((b.text for b in response.content if b.type == "text"), "")
            got = parse_clue(text, board_words)
            if isinstance(got, tuple):
                if self.cache:
                    self.cache.put(self.model_id, prompt, text, got[0], got[1], tin, tout, attempt)
                return got
            reasons.append(got)
            ask = prompt + _RETRY.format(reason=got)
        raise RuntimeError(f"{self.model_id} gave no usable clue in {ATTEMPTS} attempts: " + " | ".join(reasons))
