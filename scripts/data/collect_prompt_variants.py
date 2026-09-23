"""Does HOW we ask the teacher change what it picks? Buys the same positions
under four prompts, several samples each, so the answers can be compared.

Every listener so far was trained on one prompt: clue and number, "rank ALL
of the words", with the ranking's j-th entry read as the step-j pick. That
reading is exact for what it is -- the model writes its list one word at a
time, so entry 2 is a genuine sample of "what it names second, having already
named entry 1" -- but it is not obviously the same thing as the question play
asks at step 2, "what do you guess now that entry 1 is gone". So:

    ranked            the training prompt: clue + number, rank everything
    ranked_nonumber   the same with the number withheld
    single            one word only; step 2 re-asks with the first word
                      silently removed from the list
    single_feedback   step 2 only: the first word removed AND the prompt says
                      it was guessed and was correct -- what a real guesser
                      knows before its second guess

Step 2 is conditioned on one fixed first word per position, the first word
of the ranking already in the store (itself a `ranked` sample, and not one of
the new ones). The single prompts remove it outright; for the ranked prompts
the analysis keeps the samples whose first word happens to be it. Steps 3+
are not bought: training weights them 0.75**j and they score R2 ~0.1.

Positions are the gpt-oss holdout (collected seeds >= HOLDOUT_START), which no
listener trained on, so the incumbent listener can also be scored on each
prompt's picks.

**Why a separate store.** llm_store.db caches one answer per (model, clue,
candidates, number) and returns it on a repeat call -- the opposite of what
repeated sampling needs -- and everything that reads it treats a row as a full
ranking under the training prompt. These rows are neither, so they live in
cache/prompt_variants.db, which nothing in the pipeline reads. Raw text and
token usage are kept so parsing can be audited and cost is measured rather
than estimated.

**Temperature is set explicitly (1.0), unlike every training call.** With no
temperature in the request, which is how all of llm_store.db was bought,
DeepInfra's answers to an identical prompt come back byte-identical in runs --
measured, four sequential calls gave two identical pairs, same text and same
token count -- so "8 samples" would be far fewer independent draws and the
within-prompt agreement that serves as the noise floor would be inflated.
With temperature=1.0 every call differed. That makes these the model's own
distribution, not the near-greedy one the training labels came from.

    python scripts/data/collect_prompt_variants.py --n 5 --samples 2   # pilot
    python scripts/data/collect_prompt_variants.py --n 300 --samples 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codenames.guessers.llm import _PROMPT_TEMPLATE
from codenames.guessers.openai_compat import ATTEMPTS, RETRY_FREQUENCY_PENALTY, OpenAICompatGuesser
from codenames.listener_training import DB, DEFAULT_MODEL, board_lookup, resolve_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DB = PROJECT_ROOT / "cache" / "prompt_variants.db"
HOLDOUT_START = 1_040_000
COLLECTED = 45000

_HEAD = ('You are playing the guesser role in the board game Codenames. Your spymaster gave '
         'the clue "{clue}"{count_note}.')
_ONE = ('Respond with ONLY a JSON array containing that single word as a string, e.g. '
        '["word"] -- no other text.')
SINGLE = (_HEAD + " Here are the words still available to guess:\n\n{words}\n\n"
          "Which ONE of the words above would you guess first? " + _ONE)
SINGLE_FEEDBACK = (_HEAD + ' You already guessed "{prev}", and it was correct -- it was one of '
                   "your team's words. Here are the words still available to guess:\n\n{words}\n\n"
                   "Which ONE of the words above would you guess next? " + _ONE)


def holdout_positions(n: int, seed: int) -> list[dict]:
    """`n` holdout positions from the store, drawn reproducibly."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute("SELECT clue, candidates, number, ranking FROM responses "
                        "WHERE model=? AND number IS NOT NULL", (DEFAULT_MODEL,)).fetchall()
    conn.close()
    boards = board_lookup(0, COLLECTED)
    out = []
    for clue, cand_j, number, rank_j in rows:
        cand, rank = json.loads(cand_j), json.loads(rank_j)
        if sorted(w.lower() for w in cand) != sorted(w.lower() for w in rank):
            continue
        s = resolve_seed(rank, boards)
        if s is None or s < HOLDOUT_START:
            continue
        out.append({"seed": s, "clue": clue, "candidates": cand, "number": int(number),
                    "first": rank[0]})
    out.sort(key=lambda p: (p["seed"], p["clue"], p["number"]))
    print(f"holdout positions available: {len(out)}")
    return random.Random(seed).sample(out, min(n, len(out)))


def named_words(text: str, candidates: list[str]) -> list[str]:
    """Board words in the response's JSON array, in the model's order, deduped."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        raw = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    by_lower = {w.lower(): w for w in candidates}
    out: list[str] = []
    for x in raw:
        w = by_lower.get(str(x).strip().lower())
        if w and w not in out:
            out.append(w)
    return out


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, seed INTEGER, clue TEXT, number INTEGER,
            method TEXT, step INTEGER, sample INTEGER, candidates TEXT, removed TEXT,
            text TEXT, picks TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
            attempts INTEGER, temperature REAL)""")
        self.conn.commit()
        self.lock = threading.Lock()

    def done(self) -> set[str]:
        return {k for (k,) in self.conn.execute("SELECT key FROM calls")}

    def put(self, row: dict) -> None:
        with self.lock:
            self.conn.execute(f"INSERT OR REPLACE INTO calls ({','.join(row)}) "
                              f"VALUES ({','.join('?' * len(row))})", list(row.values()))
            self.conn.commit()


def plan(positions: list[dict], samples: int) -> list[dict]:
    jobs = []
    for p in positions:
        rest = [w for w in p["candidates"] if w != p["first"]]
        specs = [("ranked", 1, p["candidates"], []), ("ranked_nonumber", 1, p["candidates"], []),
                 ("single", 1, p["candidates"], [])]
        if p["number"] >= 2:
            specs += [("single", 2, rest, [p["first"]]), ("single_feedback", 2, rest, [p["first"]])]
        for method, step, cand, removed in specs:
            for s in range(samples):
                key = f"{p['seed']}|{p['clue']}|{p['number']}|{method}|{step}|{s}"
                jobs.append({"key": key, "seed": p["seed"], "clue": p["clue"], "number": p["number"],
                             "method": method, "step": step, "sample": s,
                             "candidates": cand, "removed": removed})
    return jobs


def prompt_for(job: dict) -> str:
    # The candidate order is the store's, identical across methods and samples,
    # so the only thing that differs between methods is the question.
    words = "\n".join(job["candidates"])
    count_note = "" if job["method"] == "ranked_nonumber" else f" for {job['number']} word(s)"
    if job["method"].startswith("ranked"):
        return _PROMPT_TEMPLATE.format(clue=job["clue"], count_note=count_note, words=words)
    if job["method"] == "single_feedback":
        return SINGLE_FEEDBACK.format(clue=job["clue"], count_note=count_note, words=words,
                                      prev=job["removed"][0])
    return SINGLE.format(clue=job["clue"], count_note=count_note, words=words)


def ask(client, model: str, effort: str, job: dict, max_tokens: int, temperature: float) -> dict:
    """Same retry policy as OpenAICompatGuesser._query; raises rather than
    recording an unusable answer."""
    need = 1 if job["method"].startswith("single") else min(3, len(job["candidates"]))
    budget, problems = max_tokens, []
    ptok = ctok = 0
    for attempt in range(1, ATTEMPTS + 1):
        req = {"model": model, "max_completion_tokens": budget, "reasoning_effort": effort,
               "temperature": temperature, "messages": [{"role": "user", "content": prompt_for(job)}]}
        if attempt > 1:
            req["frequency_penalty"] = RETRY_FREQUENCY_PENALTY
        resp = client.chat.completions.create(**req)
        if resp.usage:
            ptok += resp.usage.prompt_tokens
            ctok += resp.usage.completion_tokens
        choice = resp.choices[0]
        text = choice.message.content or ""
        picks = named_words(text, job["candidates"])
        if len(picks) >= need:
            return {"text": text, "picks": picks, "prompt_tokens": ptok,
                    "completion_tokens": ctok, "attempts": attempt}
        problems.append(f"attempt {attempt}: named {len(picks)}/{need} (finish={choice.finish_reason})")
        if attempt == 1:
            budget *= 2
    raise RuntimeError(f"{job['key']}: " + "; ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DB)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    positions = holdout_positions(args.n, args.seed)
    jobs = plan(positions, args.samples)
    store = Store(args.out)
    done = store.done()
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(positions)} positions, {len(jobs)} calls planned, {len(todo)} to buy", flush=True)

    g = OpenAICompatGuesser(model="openai/gpt-oss-120b", reasoning_effort="low")
    client = g.client
    lock = threading.Lock()
    st = {"n": 0, "fail": 0, "ptok": 0, "ctok": 0, "t0": time.time()}

    def work(job: dict) -> None:
        try:
            r = ask(client, g.model, g.reasoning_effort, job, args.max_tokens, args.temperature)
        except Exception as exc:                                   # noqa: BLE001
            with lock:
                st["fail"] += 1
                print(f"  FAIL {type(exc).__name__}: {exc}", flush=True)
            return
        store.put({"key": job["key"], "seed": job["seed"], "clue": job["clue"], "number": job["number"],
                   "method": job["method"], "step": job["step"], "sample": job["sample"],
                   "candidates": json.dumps(job["candidates"]), "removed": json.dumps(job["removed"]),
                   "text": r["text"], "picks": json.dumps(r["picks"]),
                   "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                   "attempts": r["attempts"], "temperature": args.temperature})
        with lock:
            st["n"] += 1
            st["ptok"] += r["prompt_tokens"]
            st["ctok"] += r["completion_tokens"]
            if st["n"] % 200 == 0 or st["n"] == len(todo):
                el = time.time() - st["t0"]
                print(f"  {st['n']}/{len(todo)}  fails {st['fail']}  tokens in {st['ptok']:,} "
                      f"out {st['ctok']:,}  {st['n'] / el:.1f} calls/s", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(work, todo))
    print(f"\ndone: {st['n']} calls, {st['fail']} failed, tokens in {st['ptok']:,} out {st['ctok']:,}")


if __name__ == "__main__":
    main()
