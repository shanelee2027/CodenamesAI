"""Can gpt-oss catch up on a clue it missed, if it is told the earlier clue?

Human guessers do: a word the spymaster meant for an earlier clue and the
team did not find usually gets guessed on a later turn, often with the extra
(number + 1) guess. Our gpt-oss guesser cannot, since every call sees only the
board and the current clue. This asks whether it COULD, a capability check
only (docs/log.md, "Clue history probe").

**Positions.** Recorded games (cache/llm_store.db, game_records): a team's
turn right after its own previous clue came up short (fewer own words found
than the number). At most one position per game.

**The leftover word L.** The previous clue is re-ranked by gpt-oss over the
CURRENT board (the standard prompt, a separate call): L is the highest-ranked
word of the team's still on the board. Positions are kept only when L is
ranked 1st or 2nd overall, i.e. the earlier clue clearly points at it. This
uses no listener of ours, and no condition's output.

**Conditions**, each the standard ranking prompt for the current clue:
- none: as in the arena;
- history: preceded by the previous clue, its number and what was guessed;
- history + hint: plus "a word your spymaster meant for an earlier clue may
  still be on the board";
- placebo: a previous clue from another game (no hint), to separate using
  the history from a longer prompt.

**Measured:** whether L is in the top number + 1 (the extra guess the rules
allow), its mean rank, and own words found under number + 1 (stopping at the
first word not the team's).

Raw responses of the new prompts go to table `history_probe` of
cache/llm_store.db; the standard-prompt rankings use the guesser's own cache.

    python scripts/tools/probe_clue_history.py --n 300 --max-calls 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "cache" / "llm_store.db"
CONDITIONS = ("none", "history", "history + hint", "placebo")
HINT = " A word your spymaster meant for an earlier clue may still be on the board, not yet guessed."


def describe(clue: str, number: int, guesses: list) -> str:
    """The previous turn as the guesser would remember it."""
    parts = [f"{w} (correct)" if r == "own" else f"{w} (wrong, which ended your turn)" for w, r in guesses]
    said = ", then ".join(parts) if parts else "no words"
    return (f"Earlier in this game, your spymaster's previous clue to you was \"{clue}\" for {number} "
            f"word(s). You guessed {said}.")


def positions(rng: random.Random) -> list[dict]:
    """Every eligible (game, turn), at most one per game, in random order."""
    from codenames.board import Board

    con = sqlite3.connect(DB)
    out = []
    for gid, seed, board_json, turns_json in con.execute("SELECT id, seed, board, turns FROM game_records"):
        board, turns = json.loads(board_json), json.loads(turns_json)
        words = Board.generate(seed=seed).words
        if set(words) != {w for ws in board.values() for w in ws}:
            continue
        team_words = {"A": set(board["own"]), "B": set(board["opponent"])}
        revealed, last, eligible = set(), {}, []
        for t, turn in enumerate(turns):
            T = turn["team"]
            prev = last.get(T)
            if prev is not None:
                found = sum(r == "own" for _, r in prev["guesses"])
                left = [w for w in words if w in team_words[T] and w not in revealed]
                if found < prev["number"] and left:
                    eligible.append({"game": gid, "turn": t, "team": T, "clue": turn["clue"],
                                     "number": int(turn["number"]),
                                     "candidates": [w for w in words if w not in revealed],
                                     "team_words": sorted(team_words[T] - revealed),
                                     "prev": {"clue": prev["clue"], "number": int(prev["number"]),
                                              "guesses": prev["guesses"]}})
            revealed |= {w for w, _ in turn["guesses"]}
            last[T] = turn
        if eligible:
            out.append(rng.choice(eligible))
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="positions to keep")
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    from codenames.guessers.llm import _PROMPT_TEMPLATE
    from codenames.guessers.openai_compat import ATTEMPTS, RETRY_FREQUENCY_PENALTY, OpenAICompatGuesser

    g = OpenAICompatGuesser(model="openai/gpt-oss-120b", provider="deepinfra", reasoning_effort="low", cache_path=DB)
    lock = threading.Lock()
    calls = {"n": 0}
    create = g._create

    def counted(request):
        with lock:
            if calls["n"] >= args.max_calls:
                raise RuntimeError("call cap reached")
            calls["n"] += 1
        return create(request)

    g._create = counted
    con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
    con.execute("""CREATE TABLE IF NOT EXISTS history_probe (
        cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, condition TEXT NOT NULL, clue TEXT NOT NULL,
        candidates TEXT NOT NULL, number INTEGER, prompt TEXT NOT NULL, ranking TEXT NOT NULL, raw TEXT NOT NULL)""")
    con.commit()

    def standard(clue, cands, number):
        return g._ranked(clue, list(cands), number)

    def custom(condition, prefix, clue, cands, number):
        """The standard prompt after `prefix`, validated and retried as the arena's are."""
        prompt = prefix + "\n\n" + _PROMPT_TEMPLATE.format(clue=clue, count_note=f" for {number} word(s)",
                                                           words="\n".join(cands))
        key = json.dumps([g.cache_model_id, prompt])
        with lock:
            row = con.execute("SELECT ranking FROM history_probe WHERE cache_key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        budget, problems = g.max_tokens, []
        for attempt in range(1, ATTEMPTS + 1):
            request = {"model": g.model, "max_completion_tokens": budget, "reasoning_effort": g.reasoning_effort,
                       "messages": [{"role": "user", "content": prompt}]}
            if attempt > 1:
                request["frequency_penalty"] = RETRY_FREQUENCY_PENALTY
            choice = g._create(request).choices[0]
            text = choice.message.content or ""
            named = g._named_by_model(text, cands)
            problem = g._reject(choice, text, named, cands, number)
            if problem is None:
                ranking = g._ranking_from(text, cands)
                with lock:
                    con.execute("INSERT OR REPLACE INTO history_probe VALUES (?,?,?,?,?,?,?,?,?)",
                                (key, g.cache_model_id, condition, clue, json.dumps(cands), number, prompt,
                                 json.dumps(ranking), text))
                    con.commit()
                return ranking
            problems.append(problem)
            if attempt == 1:
                budget *= 2
        raise RuntimeError("; ".join(problems))

    rng = random.Random(args.seed)
    pool = positions(rng)
    print(f"{len(pool)} eligible games; keeping up to {args.n} positions; at most {args.max_calls} calls", flush=True)

    # Phase 1: the leftover word, from the previous clue re-ranked over the current board.
    def leftover(p):
        try:
            ref = standard(p["prev"]["clue"], p["candidates"], p["prev"]["number"])
        except Exception as exc:
            return p, None, str(exc)
        mine = [w for w in ref if w in set(p["team_words"])]
        return p, (mine[0], ref.index(mine[0])) if mine else None, None

    kept, tried = [], 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        while len(kept) < args.n and tried < len(pool):
            batch = pool[tried: tried + max(2 * (args.n - len(kept)), 16)]
            tried += len(batch)
            for p, got, err in ex.map(leftover, batch):
                if got is not None and got[1] <= 1 and len(kept) < args.n:
                    kept.append({**p, "L": got[0], "L_ref_rank": got[1] + 1})
    print(f"phase 1: {tried} positions tried, {len(kept)} kept (the earlier clue ranks a team word 1st or 2nd); "
          f"{calls['n']} calls so far", flush=True)

    # Phase 2: the four conditions.
    for i, p in enumerate(kept):
        p["placebo"] = kept[(i + 1) % len(kept)]["prev"]

    def conditions(p):
        out = {}
        prev = describe(p["prev"]["clue"], p["prev"]["number"], p["prev"]["guesses"])
        plac = describe(p["placebo"]["clue"], p["placebo"]["number"], p["placebo"]["guesses"])
        for c in CONDITIONS:
            try:
                if c == "none":
                    out[c] = standard(p["clue"], p["candidates"], p["number"])
                else:
                    prefix = {"history": prev, "history + hint": prev + HINT, "placebo": plac}[c]
                    out[c] = custom(c, prefix, p["clue"], p["candidates"], p["number"])
            except Exception as exc:
                out[c] = None
                p.setdefault("errors", []).append(f"{c}: {str(exc)[:120]}")
        p["rankings"] = out
        return p

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        kept = list(ex.map(conditions, kept))
    done = [p for p in kept if all(p["rankings"][c] is not None for c in CONDITIONS)]
    print(f"phase 2: {len(done)} of {len(kept)} positions with all four rankings; {calls['n']} calls in total "
          f"(~${calls['n'] * 0.08 / 1000:.2f})\n", flush=True)

    def own_found(r, p):
        mine, n = set(p["team_words"]), 0
        for w in r[: p["number"] + 1]:
            if w not in mine:
                break
            n += 1
        return n

    print(f"{'condition':16s} {'L in top k+1':>12s} {'L first':>8s} {'L mean rank':>11s} {'own found (k+1)':>16s}")
    summary = {}
    for c in CONDITIONS:
        top = np.array([p["L"] in p["rankings"][c][: p["number"] + 1] for p in done])
        first = np.array([p["rankings"][c][0] == p["L"] for p in done])
        rank = np.array([p["rankings"][c].index(p["L"]) + 1 for p in done])
        found = np.array([own_found(p["rankings"][c], p) for p in done])
        summary[c] = {"L_top_k1": float(top.mean()), "L_first": float(first.mean()), "L_rank": float(rank.mean()),
                      "own_found": float(found.mean())}
        print(f"{c:16s} {top.mean():12.1%} {first.mean():8.1%} {rank.mean():11.2f} {found.mean():16.2f}")

    # Paired, against "none": positions where L moved into / out of the top k + 1.
    print()
    for c in CONDITIONS[1:]:
        up = sum(p["L"] in p["rankings"][c][: p["number"] + 1] and p["L"] not in p["rankings"]["none"][: p["number"] + 1]
                 for p in done)
        down = sum(p["L"] not in p["rankings"][c][: p["number"] + 1] and p["L"] in p["rankings"]["none"][: p["number"] + 1]
                   for p in done)
        print(f"  {c:16s} vs none: L entered the top k+1 on {up} positions, left it on {down}")

    print("\nExamples (history moved L up):")
    shown = 0
    for p in done:
        r0, r1 = p["rankings"]["none"], p["rankings"]["history"]
        if r1.index(p["L"]) < r0.index(p["L"]) and shown < args.examples:
            shown += 1
            print(f"  earlier \"{p['prev']['clue']}\" {p['prev']['number']} (guessed "
                  f"{', '.join(w for w, _ in p['prev']['guesses']) or 'nothing'}); now \"{p['clue']}\" {p['number']}; "
                  f"L = {p['L']}: rank {r0.index(p['L']) + 1} -> {r1.index(p['L']) + 1}; "
                  f"with history: {', '.join(r1[: p['number'] + 1])}")
    out = PROJECT_ROOT / "cache" / "report" / "clue_history_probe.json"
    out.write_text(json.dumps({"args": vars(args), "calls": calls["n"], "summary": summary, "positions": kept},
                              indent=1, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
