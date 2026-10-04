"""Which board words does a clue point at? Graded labels from an LLM, the data
for a listener that can end its turn (docs/log.md, "Relatedness labels").

**Why.** The listener has been distilled from full rankings: gpt-oss orders
all ~20 words for every clue, and the model learns where each word sits in
that order. A human guesser does not rank 25 words. They see a clue as
pointing at a few words or not, and stop when nothing is left that it points
at. The ranking data has no such stop, so a spymaster fitted to it treats
every extra guess as a free lottery ticket and announces numbers no human
would chase ("obscure" for Alien, Triangle, Spot, Sound).

**The question asked.** The clue and the board, without the number, and
each word sorted into three groups: words the clue clearly points at
("guess"), words it could plausibly point at ("stretch"), and the rest. The
answer is sets, not orders: the prompt asks for each group in board order.
Two graded levels rather than one set, so how aggressive the "related" set
is gets chosen afterwards (guess, or guess + stretch), not fixed forever by
the prompt's wording. Order within a set comes from elsewhere (the stored
gpt-oss ranking of the same position).

Answers are cached in their own table of cache/llm_store.db, keyed by model,
prompt version, clue and candidate list, so a re-run resumes and costs
nothing for what is already bought.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

ATTEMPTS = 3

_HEAD = """You are playing the guesser role in the board game Codenames. Your spymaster \
gave the clue "{clue}". Here are the words still on the board:

{words}

Sort the words into three groups:
- "guess": words you would guess for this clue, because the clue clearly points at them.
"""
_TAIL = """- every other word: the clue does not point at it, and you would not guess it.

Do not rank the words: list each group in the order the words appear above. Respond with \
ONLY a JSON object of the form {{"guess": ["word", ...], "stretch": ["word", ...]}}, with \
no other text. Either list may be empty. Every word you list must be one of the board \
words above, and no word may be in both lists."""

# Prompt versions; the version is part of every stored answer's key.
# graded-v1's "stretch" was far too loose in the pilot (mean 4.0 words, 24% with
# 6 or more: "sexual" -> Shadow, Figure, Force, Parachute, ...), so v2 asks for
# a link a reasonable player would see, as a 2nd or 3rd word.
PROMPTS = {
    "graded-v1": _HEAD + """- "stretch": words the clue could be pointing at. You would try them if your spymaster \
asked for more words than your "guess" group holds, but they are not an obvious fit.
""" + _TAIL,
    "graded-v2": _HEAD + """- "stretch": words you would still guess as a second or third word for this clue if \
your spymaster's number asked for more words than your "guess" group holds. Only words with \
a link to the clue that a reasonable player would see and accept, not any loose association.
""" + _TAIL,
}
PROMPT_VERSION = "graded-v2"


def parse(text: str, candidates: list[str]) -> tuple[list[str], list[str], int]:
    """(guess, stretch, unknown names) from a response, each list in board
    order. The last JSON object with a "guess" key is used (a model that
    corrects itself writes the final answer last). Raises ValueError if there
    is none."""
    index = {w.lower(): w for w in candidates}
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("guess"), list):
            continue
        unknown = 0
        groups = []
        for key in ("guess", "stretch"):
            names = []
            for w in obj.get(key) or []:
                hit = index.get(str(w).strip().lower())
                if hit is None:
                    unknown += 1
                elif hit not in names:
                    names.append(hit)
            groups.append(names)
        guess, stretch = groups
        stretch = [w for w in stretch if w not in guess]
        order = {w: i for i, w in enumerate(candidates)}
        return sorted(guess, key=order.get), sorted(stretch, key=order.get), unknown
    raise ValueError("no JSON object with a 'guess' list")


class RelatednessStore:
    """Answers in table `relatedness` of the response store."""

    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=60)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS relatedness (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                clue TEXT NOT NULL,
                candidates TEXT NOT NULL,
                guess TEXT NOT NULL,
                stretch TEXT NOT NULL,
                raw TEXT NOT NULL
            )""")
        self._conn.commit()

    @staticmethod
    def key(model: str, clue: str, candidates: list[str], version: str = PROMPT_VERSION) -> str:
        return json.dumps([model, version, clue, list(candidates)])

    def get(self, model: str, clue: str, candidates: list[str],
            version: str = PROMPT_VERSION) -> tuple[list[str], list[str]] | None:
        with self._lock:
            row = self._conn.execute("SELECT guess, stretch FROM relatedness WHERE cache_key = ?",
                                     (self.key(model, clue, candidates, version),)).fetchone()
        return None if row is None else (json.loads(row[0]), json.loads(row[1]))

    def put(self, model: str, clue: str, candidates: list[str], guess: list[str], stretch: list[str],
            raw: str, version: str = PROMPT_VERSION) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO relatedness VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.key(model, clue, candidates, version), model, version, clue, json.dumps(candidates),
                 json.dumps(guess), json.dumps(stretch), raw))
            self._conn.commit()


class RelatednessAsker:
    """Asks an OpenAI-compatible model (through an OpenAICompatGuesser's
    client and busy-retry) for graded labels, cached in the store."""

    def __init__(self, guesser, store: RelatednessStore, version: str = PROMPT_VERSION):
        self.g, self.store, self.version = guesser, store, version
        self.model = guesser.cache_model_id
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0, "unknown_names": 0}
        self._lock = threading.Lock()

    def ask(self, clue: str, candidates: list[str]) -> tuple[list[str], list[str]]:
        got = self.store.get(self.model, clue, candidates, self.version)
        if got is not None:
            return got
        prompt = PROMPTS[self.version].format(clue=clue, words="\n".join(candidates))
        problems = []
        for attempt in range(ATTEMPTS):
            request = {"model": self.g.model, "max_completion_tokens": self.g.max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}
            if self.g.reasoning_effort is not None:
                request["reasoning_effort"] = self.g.reasoning_effort
            resp = self.g._create(request)
            text = resp.choices[0].message.content or ""
            with self._lock:
                self.usage["calls"] += 1
                self.usage["retries"] += attempt > 0
                if resp.usage is not None:
                    self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["output_tokens"] += resp.usage.completion_tokens or 0
            try:
                guess, stretch, unknown = parse(text, candidates)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            with self._lock:
                self.usage["unknown_names"] += unknown
            self.store.put(self.model, clue, candidates, guess, stretch, text, self.version)
            return guess, stretch
        raise RuntimeError(f"no usable answer for clue {clue!r}: " + "; ".join(problems))
