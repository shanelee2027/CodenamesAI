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

For training, later steps follow the model's own most probable path: step j
conditions on its argmax at steps 1..j-1, so nothing from any other guesser
enters -- the path a greedy guesser would walk, and what the Plackett-Luce
training groups remove at each step. For evaluating the model as a predictor
of a real guesser, step j instead conditions on that guesser's actual picks.

Two interchangeable backends: `HFBackend` (transformers, many positions per
forward pass) and `VLLMBackend` (vLLM with prefix caching, in its own
environment). `score_positions` drives either in rounds, one step of every
position per batch.

`transformers` and `torch` are imported lazily: nothing else in the package
needs them, and the play path must not.
"""

from __future__ import annotations

import json
import sqlite3
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


def load_tokenizer(model: str = DEFAULT_LM):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def chat(tok, user: str) -> str:
    """The prompt as the model sees it: its chat template, thinking disabled,
    ending where the assistant's answer begins."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@dataclass
class Request:
    """One step of one position: score each remaining word's continuation."""
    remaining: list[str]
    full: list[list[int]]      # token ids of prompt + answer so far + word + WORD_END
    common: int                # tokens every candidate shares; scored from here on


def build_request(tok, clue: str, candidates: list[str], number: int | None,
                  picks: list[str]) -> Request:
    remaining = [w for w in candidates if w not in picks]
    head = chat(tok, prompt_for(clue, candidates, number)) + answer_prefix(picks)
    full = [tok(head + w + WORD_END, add_special_tokens=False).input_ids for w in remaining]
    # The shared prefix is computed from the tokenisations themselves rather
    # than from `head`, because BPE may merge the opening quote into a word.
    common = 0
    for col in zip(*full):
        if all(t == col[0] for t in col):
            common += 1
        else:
            break
    return Request(remaining, full, min(common, min(len(f) for f in full) - 1))


def normalise(req: Request, scores: np.ndarray) -> StepDistribution:
    lp = np.asarray(scores, dtype=np.float64)
    return StepDistribution(req.remaining, lp - np.logaddexp.reduce(lp))


class HFBackend:
    """transformers, batching many requests per forward pass.

    Same-length shared prefixes are prefilled together (see `score`); each
    prefix's key/value cache is then copied once per candidate
    (`batch_select_indices` with repeated indices), and every candidate's few
    remaining tokens run in one second pass. `max_rows` bounds the candidates
    per pass, which is what the copied caches cost in memory (~150 KB per
    cached token for Qwen3-8B).
    """

    def __init__(self, model: str = DEFAULT_LM, device: str = "cuda", max_rows: int = 80):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.tok = load_tokenizer(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype="auto", device_map=device)
        self.lm.eval()
        self.device = device
        self.max_rows = max_rows
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def score(self, requests: list[Request]) -> list[np.ndarray]:
        """Requests are batched only with others whose shared prefix has
        EXACTLY the same token length, so no prefix is ever padded.

        Measured: padding changes Qwen3-FP8's output. With two prefixes of
        different lengths in one batch, even the unpadded one's candidate
        log-probs moved by ~1 nat, because a padding mask switches PyTorch's
        attention kernel and this checkpoint is sensitive to it. Same-length
        batches take the unmasked path and reproduce a request scored alone
        bit for bit. Prompt lengths vary by only a few tokens, so a round of
        thousands of requests still batches well.
        """
        out: list[np.ndarray | None] = [None] * len(requests)
        by_len: dict[int, list[int]] = {}
        for i, r in enumerate(requests):
            by_len.setdefault(r.common, []).append(i)
        for idxs in by_len.values():
            chunk, rows = [], 0
            for i in idxs:
                if chunk and rows + len(requests[i].full) > self.max_rows:
                    self._run(chunk, requests, out)
                    chunk, rows = [], 0
                chunk.append(i)
                rows += len(requests[i].full)
            if chunk:
                self._run(chunk, requests, out)
        return out

    def _run(self, idxs: list[int], requests: list[Request], out: list) -> None:
        torch, dev = self.torch, self.device
        reqs = [requests[i] for i in idxs]
        prefixes = [r.full[0][:r.common] for r in reqs]
        width = max(len(p) for p in prefixes)
        ids = torch.full((len(reqs), width), self.pad, dtype=torch.long, device=dev)
        mask = torch.zeros((len(reqs), width), dtype=torch.long, device=dev)
        for b, p in enumerate(prefixes):             # left-padded: every prefix ends at `width`
            ids[b, width - len(p):] = torch.tensor(p, device=dev)
            mask[b, width - len(p):] = 1
        row_of = torch.tensor([b for b, r in enumerate(reqs) for _ in r.full], device=dev)
        tails = [f[r.common:] for r in reqs for f in r.full]
        tw = max(len(t) for t in tails)
        tids = torch.full((len(tails), tw), self.pad, dtype=torch.long, device=dev)
        tmask = torch.zeros((len(tails), tw), dtype=torch.long, device=dev)
        for i, t in enumerate(tails):
            tids[i, :len(t)] = torch.tensor(t, device=dev)
            tmask[i, :len(t)] = 1
        with torch.no_grad():
            o = self.lm(ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                        use_cache=True)
            first = o.logits[:, -1].float().log_softmax(-1)[row_of, tids[:, 0]]
            cache = o.past_key_values
            cache.batch_select_indices(row_of)
            plen = mask.sum(-1)[row_of]
            lg = self.lm(tids, past_key_values=cache, attention_mask=torch.cat([mask[row_of], tmask], 1),
                         position_ids=plen[:, None] + torch.arange(tw, device=dev)[None],
                         use_cache=False).logits.float().log_softmax(-1)
            rest = lg[:, :-1].gather(-1, tids[:, 1:, None])[..., 0] * tmask[:, 1:]
            total = (first + rest.sum(-1)).double().cpu().numpy()
        k = 0
        for i, r in zip(idxs, reqs):
            out[i] = total[k:k + len(r.full)]
            k += len(r.full)


class VLLMBackend:
    """vLLM, with automatic prefix caching: every candidate of a request is its
    own prompt, and the shared prefix is computed once and reused. The score is
    read from `prompt_logprobs`, the log-probability vLLM reports for each
    token of the prompt itself. Runs in its own environment (.venv-vllm): vLLM
    pins a torch of its own."""

    def __init__(self, model: str = DEFAULT_LM, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 2048):
        from vllm import LLM

        self.tok = load_tokenizer(model)
        self.llm = LLM(model=model, enable_prefix_caching=True, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization)

    def score(self, requests: list[Request]) -> list[np.ndarray]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=f) for r in requests for f in r.full]
        outs = self.llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        scores, k = [], 0
        for r in requests:
            s = np.empty(len(r.full))
            for c, f in enumerate(r.full):
                pl = outs[k].prompt_logprobs
                s[c] = sum(pl[i][f[i]].logprob for i in range(r.common, len(f)))
                k += 1
            scores.append(s)
        return scores


def score_positions(backend, jobs: list[tuple], own_path: bool = True) -> list[list[StepDistribution]]:
    """Distributions for many positions, in rounds: round j scores step j of
    every position still going, as one batch.

    `jobs` are (clue, candidates, number, depth, path). With `own_path`, step j
    conditions on the model's own argmax at steps 1..j-1 -- the teacher alone,
    nothing from any other guesser. Otherwise it conditions on `path`, the
    order some guesser actually picked in, which is how the model is scored as
    a predictor of THAT guesser.
    """
    results: list[list[StepDistribution]] = [[] for _ in jobs]
    picks: list[list[str]] = [[] for _ in jobs]
    depth = [min(j[3], len(j[1]) - 1) for j in jobs]
    for step in range(max(depth, default=0)):
        live = [i for i in range(len(jobs)) if step < depth[i]]
        reqs = [build_request(backend.tok, jobs[i][0], jobs[i][1], jobs[i][2], picks[i]) for i in live]
        for i, req, sc in zip(live, reqs, backend.score(reqs)):
            d = normalise(req, sc)
            results[i].append(d)
            picks[i].append(d.top if own_path else jobs[i][4][step])
    return results


class DistributionStore:
    """Local cache of teacher distributions, apart from cache/llm_store.db:
    these cost nothing to recompute, and keeping them out of the paid-response
    store means nothing here can be mistaken for a ranking an API guesser
    returned.

    `source` says what the later steps were conditioned on: "own" (the model's
    own argmax -- the teacher alone), or the id of the guesser whose actual
    pick order was used (for scoring the model as a predictor of that guesser).
    The two are never mixed: one trains a listener, the other evaluates one.
    """

    def __init__(self, path: Path = DEFAULT_DIST_DB):
        # WAL plus a busy timeout, so an analysis reading the store while a
        # collection run writes to it cannot make the writer fail on a lock.
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS dists (model TEXT, source TEXT, clue TEXT, candidates TEXT, "
            "number INT, steps TEXT, PRIMARY KEY (model, source, clue, candidates, number))")
        self.conn.commit()

    def done(self, model: str, source: str) -> set[tuple]:
        return {(c, cand, n) for c, cand, n in self.conn.execute(
            "SELECT clue, candidates, number FROM dists WHERE model=? AND source=?", (model, source))}

    def put_many(self, model: str, source: str, rows: list[tuple]) -> None:
        """rows: (clue, candidates, number, [StepDistribution, ...])."""
        self.conn.executemany("INSERT OR REPLACE INTO dists VALUES (?, ?, ?, ?, ?, ?)", [
            (model, source, clue, json.dumps(cand), number, json.dumps(
                [{"remaining": d.remaining, "logprobs": [round(float(x), 5) for x in d.logprobs]}
                 for d in steps]))
            for clue, cand, number, steps in rows])
        self.conn.commit()

    def all(self, model: str, source: str):
        """(clue, candidates, number, steps) rows, JSON-decoded."""
        rows = self.conn.execute(
            "SELECT clue, candidates, number, steps FROM dists WHERE model=? AND source=?",
            (model, source)).fetchall()
        for clue, cand, number, steps in rows:
            yield clue, json.loads(cand), number, json.loads(steps)
