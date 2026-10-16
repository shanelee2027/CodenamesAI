"""Which question gets a usable comfort signal out of gpt-oss? A pilot.

Comfort, as Shane and the professor use it, is how obvious a clue makes it
which words are meant (docs/log.md, "Comfort probe"). Asking gpt-oss "how
confident are you" measures how likely it thinks it is to be right, which is
close to the expected value the spymasters already maximise. These two
questions ask about the clue instead:

- `sure`: list the board words you are SURE the clue is meant for (fewer than
  the number, or none, is allowed), then the ones you would only be guessing
  at. Comfort = len(sure) relative to the number. The list is checkable: are
  the sure words actually the team's?
- `clarity`: how clear is it which words the clue is meant for, 1-5, each
  level described.

Each is its own call, from the guesser's seat: the board's unrevealed words,
the clue and its number, as the arena's guesser sees them. Neither shows the
intended words or any earlier answer.

**Turns.** Clues the named spymasters actually gave in their stored games on
the gpt-oss suite (holdout_v1_gptoss), sampled at random, so each turn also
has what gpt-oss did with it in the game: whether its guesses found all N.
A useful question spreads its answers across clues and tracks that outcome.

Raw responses go to table `comfort_probe` of cache/llm_store.db.

    python scripts/tools/probe_comfort.py --per-model 25 --max-calls 260
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "cache" / "llm_store.db"
SUITE = "93f5a196b26054d1"                     # holdout_v1_gptoss
MODELS = {
    "learned_listener (September incumbent)": "learned_listener:0e07db5a45d1",
    "assoc_profile_listener (incumbent)": "assoc_profile_listener:17861186a927",
    "win_prob, assoc booster": "win_prob_listener:105dc03b7f48",
    "pick_index_lookahead_listener": "pick_index_lookahead_listener:1cd80f3e8da1",
}
ATTEMPTS = 3

HEAD = """You are playing the guesser role in the board game Codenames. Your spymaster gave the \
clue "{clue}" for {k} word(s). Here are the words still available to guess:

{words}
"""
QUESTIONS = {
    "sure": """
Which of the words above are you SURE this clue is meant for? List only the words you are sure \
about -- that can be fewer than {k}, or none. Then list the words you would only be guessing at.

Respond with ONLY a JSON object of the form {{"sure": ["word", ...], "guessing": ["word", ...]}} \
-- no other text. Every word must be one of the words above.""",
    "clarity": """
How clear is it which words this clue is meant for?
1 = no idea which words are meant
2 = mostly guesswork
3 = some words are clear, the rest is guesswork
4 = fairly clear, with small doubts
5 = completely obvious

Respond with ONLY a JSON object of the form {{"clarity": <1-5>}} -- no other text.""",
}


def turns_of(con, sid: str) -> list[dict]:
    """Every turn `sid` gave in its suite games, from the guesser's seat."""
    from codenames.board import Board, load_holdout_wordlist

    vocab = load_holdout_wordlist()                # the suite's boards use the held-out words
    out = []
    for label, seed, board_json, turns_json in con.execute(
            "SELECT label, seed, board, turns FROM game_records WHERE suite_id = ? AND spymaster_id LIKE ?",
            (SUITE, f"%{sid}%")):
        seat = dict(kv.split("=", 1) for kv in label.split("|", 1)[1].split(","))
        board = json.loads(board_json)
        words = Board.generate(seed=seed, vocabulary=vocab).words
        if set(words) != {w for ws in board.values() for w in ws}:
            raise ValueError(f"seed {seed}: regenerated board differs from the stored one")
        revealed: set[str] = set()
        for t in json.loads(turns_json):
            if seat.get(t["team"]) == sid:
                own = set(board["own"] if t["team"] == "A" else board["opponent"])
                found = sum(r == "own" for _, r in t["guesses"])
                out.append({"seed": seed, "team": t["team"], "clue": t["clue"], "k": int(t["number"]),
                            "words": [w for w in words if w not in revealed],
                            "own": sorted(own - revealed), "found": found,
                            "all_found": found >= int(t["number"])})
            revealed |= {w for w, _ in t["guesses"]}
    return out


def parse(q: str, text: str, words: list[str]):
    m = list(re.finditer(r"\{.*\}", text, re.S))
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m[-1].group(0))
    if q == "clarity":
        v = int(obj["clarity"])
        if not 1 <= v <= 5:
            raise ValueError("clarity out of range")
        return v
    index = {w.lower(): w for w in words}
    sure = [index[str(w).strip().lower()] for w in obj.get("sure", []) if str(w).strip().lower() in index]
    guessing = [index[str(w).strip().lower()] for w in obj.get("guessing", []) if str(w).strip().lower() in index]
    return {"sure": list(dict.fromkeys(sure)), "guessing": list(dict.fromkeys(guessing))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-model", type=int, default=25)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", nargs="+", default=list(QUESTIONS))
    args = ap.parse_args()

    from codenames.guessers.openai_compat import OpenAICompatGuesser

    g = OpenAICompatGuesser(model="openai/gpt-oss-120b", provider="deepinfra", reasoning_effort="low")
    con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
    con.execute("""CREATE TABLE IF NOT EXISTS comfort_probe (
        cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, question TEXT NOT NULL, clue TEXT NOT NULL,
        number INTEGER, prompt TEXT NOT NULL, answer TEXT NOT NULL, raw TEXT NOT NULL)""")
    con.commit()
    rng = random.Random(args.seed)
    sample = []
    for name, sid in MODELS.items():
        ts = turns_of(con, sid)
        rng.shuffle(ts)
        sample += [{**t, "spymaster": name} for t in ts[:args.per_model]]

    lock, calls = threading.Lock(), {"n": 0, "failed": 0}

    def ask(q: str, t: dict):
        prompt = HEAD.format(clue=t["clue"], k=t["k"], words="\n".join(t["words"])) + QUESTIONS[q].format(k=t["k"])
        key = json.dumps([g.cache_model_id, prompt])
        with lock:
            row = con.execute("SELECT raw FROM comfort_probe WHERE cache_key = ?", (key,)).fetchone()
        if row:
            return parse(q, row[0], t["words"])
        for _ in range(ATTEMPTS):
            with lock:
                if calls["n"] >= args.max_calls:
                    raise RuntimeError("call cap reached")
                calls["n"] += 1
            req = {"model": g.model, "max_completion_tokens": g.max_tokens, "reasoning_effort": "low",
                   "messages": [{"role": "user", "content": prompt}]}
            text = g._create(req).choices[0].message.content or ""
            try:
                ans = parse(q, text, t["words"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            with lock:
                con.execute("INSERT OR REPLACE INTO comfort_probe VALUES (?,?,?,?,?,?,?,?)",
                            (key, g.cache_model_id, q, t["clue"], t["k"], prompt, json.dumps(ans), text))
                con.commit()
            return ans
        with lock:
            calls["failed"] += 1
        raise RuntimeError("no parsable answer")

    def run(t):
        t["answers"] = {}
        for q in args.questions:
            try:
                t["answers"][q] = ask(q, t)
            except Exception as exc:                   # noqa: BLE001
                t["answers"][q] = None
                t.setdefault("errors", []).append(f"{q}: {str(exc)[:100]}")
        return t

    print(f"{len(sample)} turns ({args.per_model} per model), questions {args.questions}, "
          f"at most {args.max_calls} calls", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        sample = list(ex.map(run, sample))
    done = [t for t in sample if all(t["answers"].get(q) is not None for q in args.questions)]
    print(f"{len(done)} turns with every answer; {calls['n']} calls (~${calls['n'] * 0.07 / 1000:.2f}), "
          f"{calls['failed']} unanswerable\n")

    y = np.array([t["all_found"] for t in done], dtype=float)

    def auc(score):
        s = np.asarray(score, dtype=float)
        pos, neg = s[y == 1], s[y == 0]
        return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))

    summary = {}
    if "sure" in args.questions:
        n_sure = np.array([len(t["answers"]["sure"]["sure"]) for t in done], dtype=float)
        k = np.array([t["k"] for t in done], dtype=float)
        ratio = np.minimum(n_sure / k, 1.0)
        own_hit = [w in t["own"] for t in done for w in t["answers"]["sure"]["sure"]]
        print("sure: number of words it is sure about, against the clue's number")
        print("   " + "  ".join(f"k={kk}: " + ", ".join(f"{v}:{c}" for v, c in sorted(Counter(
            int(x) for x, kx in zip(n_sure, k) if kx == kk).items())) for kk in sorted(set(k.astype(int)))))
        print(f"   share of sure words that are the team's: {np.mean(own_hit):.1%} ({len(own_hit)} words)")
        print(f"   tracks the game (all N found): AUC of sure/number {auc(ratio):.3f}; "
              f"sure >= number on {np.mean(n_sure >= k):.0%} of turns")
        summary["sure"] = {"precision": float(np.mean(own_hit)), "auc": auc(ratio),
                           "sure_ge_k": float(np.mean(n_sure >= k))}
    if "clarity" in args.questions:
        c = np.array([t["answers"]["clarity"] for t in done], dtype=float)
        print("\nclarity: distribution " + ", ".join(f"{v}:{n}" for v, n in sorted(Counter(c.astype(int)).items())))
        print(f"   tracks the game (all N found): AUC {auc(c):.3f}")
        summary["clarity"] = {"dist": dict(Counter(c.astype(int).tolist())), "auc": auc(c)}
    print(f"\n   base rate: gpt-oss found all N on {y.mean():.0%} of these turns")

    print("\nBy spymaster (pilot sizes; not a comparison)")
    for name in MODELS:
        ts = [t for t in done if t["spymaster"] == name]
        if not ts:
            continue
        line = f"   {name:40s} n={len(ts):3d}  mean k {np.mean([t['k'] for t in ts]):.2f}"
        if "sure" in args.questions:
            line += f"  sure/k {np.mean([min(len(t['answers']['sure']['sure']) / t['k'], 1) for t in ts]):.2f}"
        if "clarity" in args.questions:
            line += f"  clarity {np.mean([t['answers']['clarity'] for t in ts]):.2f}"
        line += f"  all found {np.mean([t['all_found'] for t in ts]):.0%}"
        print(line)

    out = PROJECT_ROOT / "cache" / "report" / "comfort_probe_pilot.json"
    out.write_text(json.dumps({"args": vars(args), "calls": calls["n"], "summary": summary, "turns": sample},
                              indent=1, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
