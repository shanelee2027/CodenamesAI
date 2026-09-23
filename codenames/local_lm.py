"""Exact choice probabilities from a local language model, as a teacher.

An API guesser returns one sampled ranking per call: one draw from a
distribution nobody sees. A model run locally exposes the distribution
itself. Asked the same question every API guesser is asked
(`codenames/guessers/llm.py::_PROMPT_TEMPLATE`, same candidate order), the
answer is a JSON array, so the probability that the guesser's first pick is
word `w` is the probability of the continuation `["w"`, and its j-th pick,
given picks p1..p(j-1), is that of `["p1", ..., "w"`. Scoring every candidate
this way and normalising over the candidates gives the full distribution at
each step: every word's probability, not just which one was drawn.

Each candidate is scored as a whole string -- the sum of its token
log-probabilities up to and including the separator that closes it, `",` --
so a word that is a prefix of another ("Car", "Carrot") is not credited with
the longer word's mass, and a word split into several tokens is not penalised
relative to a one-token word by anything but its actual probability. The
separator matters: Qwen's tokenizer writes the closing quote and comma as ONE
token, so a bare `"` is a sequence the model almost never emits, and scoring
it cost every candidate ~30 nats of noise.

Later steps follow the model's own most probable path: step j conditions on
its argmax at steps 1..j-1. That is the path a greedy guesser would walk, and
it is what the Plackett-Luce training groups remove at each step.

`transformers` and `torch` are imported lazily: nothing else in the package
needs them, and the play path must not.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codenames.guessers.llm import _PROMPT_TEMPLATE

DEFAULT_LM = "Qwen/Qwen3-8B-FP8"
DEFAULT_DIST_DB = Path(__file__).resolve().parent.parent / "cache" / "lm_distributions.db"


# What closes a pick that is not the last one, tokenised as one token.
WORD_END = '",'


def prompt_for(clue: str, candidates: list[str], number: int | None) -> str:
    count_note = f" for {number} word(s)" if number else ""
    return _PROMPT_TEMPLATE.format(clue=clue, count_note=count_note, words="\n".join(candidates))


def answer_prefix(picks: list[str]) -> str:
    """The JSON answer up to the opening quote of the next pick."""
    return '["' + "".join(f'{p}", "' for p in picks)


@dataclass
class StepDistribution:
    remaining: list[str]
    logprobs: np.ndarray       # log P(word | prefix), normalised over `remaining`

    @property
    def probs(self) -> np.ndarray:
        return np.exp(self.logprobs)

    @property
    def top(self) -> str:
        return self.remaining[int(np.argmax(self.logprobs))]


class LocalLMScorer:
    """Scores every candidate's continuation in one batched forward pass.

    The prompt and answer prefix are shared by every candidate, so they are
    run once and their key/value cache is reused for the whole batch of
    candidate suffixes -- only the few tokens of each word are recomputed.
    """

    def __init__(self, model: str = DEFAULT_LM, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype="auto", device_map=device)
        self.lm.eval()
        self.device = device
        self._lock = threading.Lock()

    def _chat(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def _tokens(self, clue: str, candidates: list[str], number: int | None,
                picks: list[str]) -> tuple[list[str], list[list[int]], int]:
        """(remaining words, each one's full token ids, shared token prefix)."""
        remaining = [w for w in candidates if w not in picks]
        head = self._chat(prompt_for(clue, candidates, number)) + answer_prefix(picks)
        full = [self.tok(head + w + WORD_END, add_special_tokens=False).input_ids for w in remaining]
        # The shared prefix is computed from the tokenisations themselves rather
        # than from `head`, because BPE may merge the opening quote into a word.
        common = 0
        for col in zip(*full):
            if all(t == col[0] for t in col):
                common += 1
            else:
                break
        return remaining, full, min(common, min(len(f) for f in full) - 1)

    def _score_tails(self, base_cache, prefix_logprobs, common: int, full: list[list[int]]) -> np.ndarray:
        """Sum of log-probs of each candidate's tokens after the shared prefix,
        given the prefix's key/value cache (cropped to `common`) and the
        next-token log-probs at its last position."""
        import copy

        torch = self.torch
        cache = copy.deepcopy(base_cache)
        cache.crop(common - cache.get_seq_length())   # negative: drop tokens past `common`
        tails = [f[common:] for f in full]
        width = max(len(t) for t in tails)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(tails), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(tails), common + width), dtype=torch.long, device=self.device)
        mask[:, :common] = 1
        for i, t in enumerate(tails):
            ids[i, :len(t)] = torch.tensor(t, device=self.device)
            mask[i, common:common + len(t)] = 1
        cache.batch_repeat_interleave(len(tails))
        pos = torch.arange(common, common + width, device=self.device).unsqueeze(0).expand(len(tails), -1)
        logits = self.lm(ids, past_key_values=cache, attention_mask=mask,
                         position_ids=pos, use_cache=False).logits.float().log_softmax(-1)
        first = prefix_logprobs[torch.tensor([t[0] for t in tails], device=self.device)]
        rest = torch.zeros(len(tails), device=self.device)
        for i, t in enumerate(tails):
            if len(t) > 1:
                rest[i] = logits[i, torch.arange(len(t) - 1, device=self.device),
                                 torch.tensor(t[1:], device=self.device)].sum()
        return (first + rest).double().cpu().numpy()

    def path_distributions(self, clue: str, candidates: list[str], number: int | None,
                           path: list[str], depth: int) -> list[StepDistribution]:
        """Steps 1..depth, step j conditioned on `path[:j-1]` -- the order some
        guesser actually picked in. One prefill serves every step: each step's
        prompt is a prefix of the next one's, so its cache is cropped, not
        recomputed."""
        torch = self.torch
        depth = min(depth, len(candidates) - 1, len(path) + 1)
        steps = [self._tokens(clue, candidates, number, path[:j]) for j in range(depth)]
        longest = max(steps, key=lambda st: st[2])
        prefix_ids = longest[1][0][:longest[2]]
        out = []
        with self._lock, torch.no_grad():
            pre = self.lm(torch.tensor([prefix_ids], device=self.device), use_cache=True)
            base, pre_logits = pre.past_key_values, pre.logits[0]
            for remaining, full, common in steps:
                if full[0][:common] != prefix_ids[:common]:
                    raise RuntimeError("a step's prompt is not a token prefix of the next; "
                                       "cannot share one prefill")
                lp = self._score_tails(base, pre_logits[common - 1].float().log_softmax(-1), common, full)
                lp -= np.logaddexp.reduce(lp)
                out.append(StepDistribution(remaining, lp))
        return out

    def step_logprobs(self, clue: str, candidates: list[str], number: int | None,
                      picks: list[str]) -> StepDistribution:
        """log P(next pick = w) for every w in `candidates` not in `picks`."""
        return self.path_distributions(clue, candidates, number, picks, len(picks) + 1)[-1]

    def distributions(self, clue: str, candidates: list[str], number: int | None,
                      depth: int) -> list[StepDistribution]:
        """Steps 1..depth, each conditioned on the model's own argmax so far."""
        picks: list[str] = []
        out = []
        for _ in range(min(depth, len(candidates) - 1)):
            d = self.step_logprobs(clue, candidates, number, picks)
            out.append(d)
            picks.append(d.top)
        return out


class DistributionStore:
    """Local, append-only cache of teacher distributions, apart from
    cache/llm_store.db: these cost nothing to recompute, and keeping them out
    of the paid-response store means nothing here can be mistaken for a
    ranking an API guesser returned."""

    def __init__(self, path: Path = DEFAULT_DIST_DB):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS dists (model TEXT, clue TEXT, candidates TEXT, number INT, "
            "path TEXT, steps TEXT, PRIMARY KEY (model, clue, candidates, number, path))")
        self.conn.commit()

    def has(self, model: str, clue: str, candidates: list[str], number: int | None,
            path: list[str]) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM dists WHERE model=? AND clue=? AND candidates=? AND number IS ? AND path=?",
            (model, clue, json.dumps(candidates), number, json.dumps(path))).fetchone() is not None

    def put(self, model: str, clue: str, candidates: list[str], number: int | None,
            path: list[str], steps: list[StepDistribution]) -> None:
        payload = [{"remaining": d.remaining, "logprobs": [round(float(x), 5) for x in d.logprobs]}
                   for d in steps]
        self.conn.execute("INSERT OR REPLACE INTO dists VALUES (?, ?, ?, ?, ?, ?)",
                          (model, clue, json.dumps(candidates), number, json.dumps(path),
                           json.dumps(payload)))
        self.conn.commit()

    def all(self, model: str):
        """(clue, candidates, number, path, steps) rows, JSON-decoded."""
        for clue, cand, number, path, steps in self.conn.execute(
                "SELECT clue, candidates, number, path, steps FROM dists WHERE model=?", (model,)):
            yield clue, json.loads(cand), number, json.loads(path), json.loads(steps)
