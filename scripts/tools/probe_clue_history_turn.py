"""Does gpt-oss use earlier clues if invited (not told) to? The turn-level
follow-up to probe_clue_history.py (docs/log.md, "Clue history probe").

The first probe asked for the arena's ranking "for this clue", which makes
earlier clues off-topic by construction, and found no effect. Here the
guesser is asked for its TURN: up to k + 1 guesses in order (the rules' extra
guess), stopping when it likes. History is every earlier clue of its own
spymaster in that game, with what was found and how many words each still
owes, plus a permission: "you can use your guesses on them if you think
that's best". Nothing says it must.

Same 298 positions and leftover word L as the first probe
(cache/report/clue_history_probe.json). Arms:
- turn: no history;
- turn + history: the history and the permission;
- turn + placebo: another game's history, same wording;
- structured + history: the answer also names, for each earlier clue with
  words unfound, the board word it now thinks was meant, or null;
- structured + placebo.

Raw responses go to table `history_probe` of cache/llm_store.db.

    python scripts/tools/probe_clue_history_turn.py --max-calls 1700
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "cache" / "llm_store.db"
POSITIONS = PROJECT_ROOT / "cache" / "report" / "clue_history_probe.json"
ARMS = ("turn", "turn + history", "turn + placebo", "structured + history", "structured + placebo")
ATTEMPTS = 3

HEAD = """You are playing the guesser role in the board game Codenames. Your spymaster gave the \
clue "{clue}" for {k} word(s). Here are the words still available to guess:

{words}
"""
RULES = """
You may make up to {k1} guesses this turn (one more than the number, as the rules allow), in order, \
and you may stop earlier. Your turn also ends at the first word that is not your team's."""
PERMISSION = (" Words meant for earlier clues may still be on the board; you can use your guesses on them "
              "if you think that's best.")
PLAIN_OUT = """

Respond with ONLY a JSON array of the words you guess, in order, e.g. ["word1", "word2"] -- no \
other text. Every word must be one of the words above."""
STRUCTURED_OUT = """

Respond with ONLY a JSON object of the form {"earlier": {"<earlier clue>": "<word>" or null, ...}, \
"guesses": ["word1", "word2", ...]} -- no other text. In "earlier", give each earlier clue that still \
has words not found, with the board word above you now think it was meant for, or null if none. \
"guesses" is your guesses this turn, in order. Every word must be one of the words above."""


def history_lines(turns: list[dict], team: str, upto: int) -> list[dict]:
    """The team's own earlier clues before turn `upto`: clue, number, found words, still owed."""
    out = []
    for turn in turns[:upto]:
        if turn["team"] != team:
            continue
        found = [w for w, r in turn["guesses"] if r == "own"]
        out.append({"clue": turn["clue"], "number": int(turn["number"]), "found": found,
                    "owed": max(0, int(turn["number"]) - len(found))})
    return out


def history_block(lines: list[dict]) -> str:
    rows = []
    for h in lines:
        got = f"you found {', '.join(h['found'])}" if h["found"] else "you found none"
        rows.append(f"- \"{h['clue']}\" for {h['number']}: {got}. "
                    + (f"{h['owed']} not found yet." if h["owed"] else "All found."))
    return "\nYour spymaster's earlier clues this game:\n" + "\n".join(rows) + "\n"


def parse(text: str, cands: list[str], k1: int, structured: bool):
    """(guesses, earlier) or raise ValueError. Guesses: board words, deduped,
    at most k + 1, in order; unknown names dropped."""
    index = {w.lower(): w for w in cands}
    pattern = r"\{.*\}" if structured else r"\[[^\[\]]*\]"
    for m in reversed(list(re.finditer(pattern, text, re.S))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        raw = obj.get("guesses") if structured and isinstance(obj, dict) else obj
        if not isinstance(raw, list):
            continue
        out = []
        for w in raw:
            hit = index.get(str(w).strip().lower())
            if hit and hit not in out:
                out.append(hit)
        earlier = None
        if structured:
            e = obj.get("earlier")
            earlier = {str(c): (index.get(str(v).strip().lower()) if v else None)
                       for c, v in e.items()} if isinstance(e, dict) else {}
        return out[:k1], earlier
    raise ValueError("no parsable answer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    from codenames.guessers.openai_compat import OpenAICompatGuesser

    g = OpenAICompatGuesser(model="openai/gpt-oss-120b", provider="deepinfra", reasoning_effort=args.effort)
    model_id = f"{g.cache_model_id}" if args.effort == "low" else f"{g.cache_model_id}@{args.effort}"
    lock = threading.Lock()
    calls = {"n": 0, "failed": 0}
    con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
    con.execute("""CREATE TABLE IF NOT EXISTS history_probe (
        cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, condition TEXT NOT NULL, clue TEXT NOT NULL,
        candidates TEXT NOT NULL, number INTEGER, prompt TEXT NOT NULL, ranking TEXT NOT NULL, raw TEXT NOT NULL)""")
    con.commit()

    games = {gid: json.loads(t) for gid, t in sqlite3.connect(DB).execute("SELECT id, turns FROM game_records")}
    pos = [p for p in json.loads(POSITIONS.read_text())["positions"]
           if all(p.get("rankings", {}).get(c) is not None for c in ("none",))]
    for p in pos:
        p["history"] = history_lines(games[p["game"]], p["team"], p["turn"])
    for i, p in enumerate(pos):
        p["placebo_history"] = pos[(i + 1) % len(pos)]["history"]

    def ask(arm: str, p: dict):
        k1 = p["number"] + 1
        structured = arm.startswith("structured")
        prompt = HEAD.format(clue=p["clue"], k=p["number"], words="\n".join(p["candidates"]))
        if "history" in arm:
            prompt += history_block(p["history"])
        elif "placebo" in arm:
            prompt += history_block(p["placebo_history"])
        prompt += RULES.format(k1=k1) + (PERMISSION if arm != "turn" else "")
        prompt += STRUCTURED_OUT if structured else PLAIN_OUT
        key = json.dumps([model_id, prompt])
        with lock:
            row = con.execute("SELECT raw FROM history_probe WHERE cache_key = ?", (key,)).fetchone()
        if row:
            return parse(row[0], p["candidates"], k1, structured)
        for attempt in range(ATTEMPTS):
            with lock:
                if calls["n"] >= args.max_calls:
                    raise RuntimeError("call cap reached")
                calls["n"] += 1
            request = {"model": g.model, "max_completion_tokens": g.max_tokens, "reasoning_effort": args.effort,
                       "messages": [{"role": "user", "content": prompt}]}
            text = g._create(request).choices[0].message.content or ""
            try:
                got = parse(text, p["candidates"], k1, structured)
            except ValueError:
                continue
            with lock:
                con.execute("INSERT OR REPLACE INTO history_probe VALUES (?,?,?,?,?,?,?,?,?)",
                            (key, model_id, arm, p["clue"], json.dumps(p["candidates"]), p["number"], prompt,
                             json.dumps(got[0]), text))
                con.commit()
            return got
        with lock:
            calls["failed"] += 1
        raise RuntimeError("no parsable answer")

    def run(p):
        p["turn_answers"] = {}
        for arm in args.arms:
            try:
                p["turn_answers"][arm] = ask(arm, p)
            except Exception as exc:
                p["turn_answers"][arm] = None
                p.setdefault("turn_errors", []).append(f"{arm}: {str(exc)[:100]}")
        return p

    print(f"{len(pos)} positions, arms {args.arms}, effort {args.effort}, at most {args.max_calls} calls", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        pos = list(ex.map(run, pos))
    done = [p for p in pos if all(p["turn_answers"].get(a) is not None for a in args.arms)]
    print(f"{len(done)} positions with every arm; {calls['n']} calls (~${calls['n'] * 0.07 / 1000:.2f}), "
          f"{calls['failed']} unanswerable\n", flush=True)

    def own_found(gs, p):
        mine, n = set(p["team_words"]), 0
        for w in gs:
            if w not in mine:
                break
            n += 1
        return n

    print(f"{'arm':22s} {'L guessed':>9s} {'L first':>8s} {'guesses':>8s} {'stops early':>11s} {'own found':>9s} "
          f"{'names L for its clue':>20s}")
    summary = {}
    for a in args.arms:
        gs = [p["turn_answers"][a][0] for p in done]
        lg = np.mean([p["L"] in x for p, x in zip(done, gs)])
        lf = np.mean([bool(x) and x[0] == p["L"] for p, x in zip(done, gs)])
        n = np.mean([len(x) for x in gs])
        early = np.mean([len(x) < p["number"] + 1 for p, x in zip(done, gs)])
        found = np.mean([own_found(x, p) for p, x in zip(done, gs)])
        named = ""
        if a.startswith("structured"):
            e = [p["turn_answers"][a][1] or {} for p in done]
            named = f"{np.mean([ei.get(p['prev']['clue']) == p['L'] for p, ei in zip(done, e)]):20.1%}"
        summary[a] = {"L_guessed": float(lg), "L_first": float(lf), "n_guesses": float(n), "stops_early": float(early),
                      "own_found": float(found)}
        print(f"{a:22s} {lg:9.1%} {lf:8.1%} {n:8.2f} {early:11.1%} {found:9.2f} {named}")

    print()
    for a in args.arms[1:]:
        ref = "turn" if a.startswith("turn") else "turn"
        up = sum(p["L"] in p["turn_answers"][a][0] and p["L"] not in p["turn_answers"][ref][0] for p in done)
        down = sum(p["L"] not in p["turn_answers"][a][0] and p["L"] in p["turn_answers"][ref][0] for p in done)
        print(f"  {a:22s} vs {ref}: L newly guessed on {up} positions, dropped on {down}")

    if "turn + history" in args.arms:
        print("\nExamples (L guessed with history, not without):")
        shown = 0
        for p in done:
            h, n0 = p["turn_answers"]["turn + history"][0], p["turn_answers"]["turn"][0]
            if p["L"] in h and p["L"] not in n0 and shown < args.examples:
                shown += 1
                print(f"  earlier \"{p['prev']['clue']}\" {p['prev']['number']}, now \"{p['clue']}\" {p['number']}, "
                      f"L = {p['L']}:  no history {n0}  ->  history {h}")
    out = PROJECT_ROOT / "cache" / "report" / f"clue_history_probe_turn_{args.effort}.json"
    out.write_text(json.dumps({"args": vars(args), "calls": calls["n"], "summary": summary, "positions": pos},
                              indent=1, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
