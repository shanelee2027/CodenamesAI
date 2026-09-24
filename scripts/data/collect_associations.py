"""Free-association lists from the teacher, one clue at a time, no board shown.

The listener's softmax runs over the board, so its training signal is blind to
one number per clue: add a constant to every word's score and nothing changes.
A clue that points strongly at one word and weakly at two (4, 2, 2) and one
that points weakly at one and not at all at the others (2, 0, 0) look the same.

These lists are a target that is NOT normalised over the board. The question
is "what does this clue make you think of?", answered over the whole
vocabulary, and a board word's count -- how many of R independent lists name
it -- is an absolute rate: high for a clue that evokes it, near zero for a
vague one, whatever else is on the board. docs/log.md has how it enters
training (a Poisson term, the Bregman divergence for counts).

The clue is shown alone on purpose. Showing the board would make the answer
board-relative again, which is the property being escaped. The cost is that
the clue may be read in a sense the board rules out, which is why this is an
extra term alongside the board-choice loss and never a replacement for it.

Temperature is set to 1.0 explicitly: without it DeepInfra returns
near-identical answers to an identical prompt (see collect_prompt_variants.py),
and R copies of one list would count as R independent draws.

Raw lists are stored and matched to board words at training time, so the
matching rule can change without buying anything again. Stored in
cache/associations.db; nothing touches llm_store.db.

    python scripts/data/collect_associations.py --limit 20 --samples 2   # pilot
    python scripts/data/collect_associations.py --samples 5
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codenames.guessers.openai_compat import ATTEMPTS, RETRY_FREQUENCY_PENALTY, OpenAICompatGuesser
from codenames.listener_training import DB, DEFAULT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DB = PROJECT_ROOT / "cache" / "associations.db"
N_WORDS = 25

PROMPT = ('List the {n} words that the word "{clue}" most strongly makes you think of, strongest '
          "association first. Each entry should be a single common word or a short name. "
          "Respond with ONLY a JSON array of strings -- no other text.")


def teacher_clues() -> list[str]:
    """Every clue the teacher was asked about, which covers the decoy boards too."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    clues = sorted({c for (c,) in conn.execute(
        "SELECT DISTINCT clue FROM responses WHERE model=? AND number IS NOT NULL", (DEFAULT_MODEL,))})
    conn.close()
    return clues


def parse(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        raw = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for x in raw:
        w = str(x).strip()
        if w and w.lower() not in (o.lower() for o in out):
            out.append(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N clues only (pilot)")
    ap.add_argument("--out", type=Path, default=OUT_DB)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    clues = teacher_clues()
    if args.limit:
        clues = clues[:: max(1, len(clues) // args.limit)][: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.out, check_same_thread=False, timeout=30)
    conn.execute("""CREATE TABLE IF NOT EXISTS lists (
        key TEXT PRIMARY KEY, clue TEXT, sample INTEGER, n_asked INTEGER, text TEXT,
        words TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, attempts INTEGER,
        temperature REAL)""")
    conn.commit()
    done = {k for (k,) in conn.execute("SELECT key FROM lists")}
    jobs = [(c, s) for c in clues for s in range(args.samples) if f"{c}|{s}" not in done]
    print(f"{len(clues)} clues x {args.samples} samples, {len(jobs)} to buy", flush=True)

    g = OpenAICompatGuesser(model="openai/gpt-oss-120b", reasoning_effort="low")
    client = g.client
    lock = threading.Lock()
    st = {"n": 0, "fail": 0, "ptok": 0, "ctok": 0, "t0": time.time()}

    def work(job: tuple[str, int]) -> None:
        clue, s = job
        ptok = ctok = 0
        budget, words, text = 2048, [], ""
        for attempt in range(1, ATTEMPTS + 1):
            req = {"model": g.model, "max_completion_tokens": budget, "reasoning_effort": g.reasoning_effort,
                   "temperature": args.temperature,
                   "messages": [{"role": "user", "content": PROMPT.format(n=N_WORDS, clue=clue)}]}
            if attempt > 1:
                req["frequency_penalty"] = RETRY_FREQUENCY_PENALTY
            try:
                resp = client.chat.completions.create(**req)
            except Exception as exc:                               # noqa: BLE001
                with lock:
                    print(f"  {clue}|{s}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if resp.usage:
                ptok += resp.usage.prompt_tokens
                ctok += resp.usage.completion_tokens
            text = resp.choices[0].message.content or ""
            words = parse(text)
            # Half the requested length is the bar: a short but real list is
            # data, an empty or degenerate one is not.
            if len(words) >= N_WORDS // 2:
                break
            if attempt == 1:
                budget *= 2
        with lock:
            if len(words) < N_WORDS // 2:
                st["fail"] += 1
                print(f"  FAIL {clue}|{s}: {len(words)} words", flush=True)
                return
            conn.execute("INSERT OR REPLACE INTO lists VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (f"{clue}|{s}", clue, s, N_WORDS, text, json.dumps(words), ptok, ctok,
                          attempt, args.temperature))
            conn.commit()
            st["n"] += 1
            st["ptok"] += ptok
            st["ctok"] += ctok
            if st["n"] % 500 == 0 or st["n"] == len(jobs):
                el = time.time() - st["t0"]
                print(f"  {st['n']}/{len(jobs)}  fails {st['fail']}  tokens in {st['ptok']:,} "
                      f"out {st['ctok']:,}  {st['n'] / el:.1f} calls/s", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(work, jobs))
    print(f"\ndone: {st['n']} lists, {st['fail']} failed, tokens in {st['ptok']:,} out {st['ctok']:,}")


if __name__ == "__main__":
    main()
