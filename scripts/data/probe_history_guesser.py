"""Does a guesser that is shown the game so far go back to words its team's
earlier clues pointed at? And does one told the score keep guessing when it
must? A small paid probe with gpt-oss, before building a history-aware
spymaster (docs/log.md, "How often spymasters re-clue their own pending words").

**Why.** The incumbent re-clues pending own words (words an earlier clue of
its team was meant for, still unfound) on 74% of its later clues
(scripts/tools/reclue_rate.py). Under the arena's guesser that is right: it
sees only the current clue, so a pending word is never found unless it is
clued again. A human remembers the earlier clues and spends spare guesses
on them. Whether a spymaster should stop re-clueing depends on how much a
guesser that sees the history actually does that.

**Positions.** The comfort arena's recorded games (holdout_v1_gptoss_subset_guess),
at the incumbent's (assoc_profile_listener) clues:
- *main*: clues after its first, with pending own words, whose own targets
  include none of them, so a pending word picked is not explained by the
  current clue. At most one per game, `--n-main` of them.
- *desperate*: the opponent has one word left and we have three or more,
  the position of the FRY 1 question (docs/log.md). `--n-desperate`.
Targets and pending words are recomputed as reclue_rate.py does (the
spymaster's listener). They are used only to score answers: the guesser
never sees them, only what a player sees.

**The prompt (turn-v1)** asks for the whole turn: the clue and its number,
the words on the board, the rules (up to number + 1 guesses, a wrong guess
ends the turn, stop at any point), and the words it would guess in order.
It does not say that guesses may go to earlier clues' words: that is the
behaviour being measured, so the prompt must not suggest it. Arms differ
only in two optional blocks:
- `none`: no history, no score;
- `history`: the game so far, every clue of both teams in order with what
  was guessed and what each guess turned out to be (public information);
- `history_score`: and a line with both teams' words left.
main runs all three, desperate `history` and `history_score`.

Answers are stored in table `turn_probe` of cache/llm_store.db, keyed by
model, prompt version and the prompt itself, so a re-run costs nothing.

    python scripts/data/probe_history_guesser.py              # ~1,100 calls, ~$0.10
    python scripts/data/probe_history_guesser.py --dry-run    # positions and one prompt, no calls
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, OpponentBoardView, Role, load_holdout_wordlist  # noqa: E402

SUITE = "dc9083567bfcf453"
INCUMBENT = "assoc_profile_listener"
VERSION = "turn-v1"
MODEL = "openai/gpt-oss-120b"
ATTEMPTS = 3

SEEN = {("own", True): "your word", ("own", False): "their word", ("opponent", True): "their word",
        ("opponent", False): "your word", ("neutral", True): "a bystander", ("neutral", False): "a bystander",
        ("assassin", True): "the assassin", ("assassin", False): "the assassin"}

PROMPT = """You are playing the guesser role in the board game Codenames.{history}
Your spymaster now gives the clue "{clue}" for {number} word(s). Here are the words still on the board:

{words}
{score}
You may make up to {most} guesses this turn, one at a time. A guess that is not one of your \
team's words ends the turn, and you may stop guessing at any point.

List the words you would guess, in the order you would guess them, and stop where you would \
stop. Respond with ONLY a JSON array of words, e.g. ["word1", "word2"], with no other text. \
The array may be empty."""


def history_block(past: list[dict], us: str) -> str:
    if not past:
        return ""
    lines = []
    for n, t in enumerate(past, 1):
        mine = t["team"] == us
        who = "Your spymaster" if mine else "The other team's spymaster"
        team = "Your team" if mine else "They"
        got = ", ".join(f"{w} ({SEEN[(r, mine)]})" for w, r in t["guesses"])
        lines.append(f'{n}. {who}: "{t["clue"]}" for {t["number"]}. '
                     + (f"{team} guessed {got}." if got else f"{team} guessed nothing."))
    return "\n\nThe game so far, in order:\n" + "\n".join(lines) + "\n"


def prompt_for(pos: dict, arm: str) -> str:
    score = (f"\nYour team has {pos['own_left']} word(s) left to find; the other team has "
             f"{pos['opp_left']}.\n") if arm == "history_score" else ""
    return PROMPT.format(history=history_block(pos["past"], pos["team"]) if arm != "none" else "",
                         clue=pos["clue"], number=pos["number"], words="\n".join(pos["words"]),
                         score=score, most=pos["number"] + 1)


def positions(db: Path) -> list[dict]:
    """Every incumbent clue in the comfort arena, with what a player sees and
    the incumbent's targets and pending words."""
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
    import reclue_rate

    reclue_rate._SIMS = SimilarityTensor.load()
    cls, kw = spymaster_spec(INCUMBENT)
    sm = cls(**kw)
    vocab = load_holdout_wordlist()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT id, seed, label, turns FROM game_records WHERE suite_id = ?", (SUITE,)).fetchall()
    conn.close()
    out = []
    for game_id, seed, label, turns_json in rows:
        seats = dict(s.split("=", 1) for s in label.split("|", 1)[1].split(","))
        us = next(t for t, name in seats.items() if name.startswith(INCUMBENT + ":"))
        board = Board.generate(seed, vocabulary=vocab)
        turns = json.loads(turns_json)
        sent: list[list[str]] = []
        for i, t in enumerate(turns):
            if t["team"] == us and t["clue"]:
                view = board if us == "A" else OpponentBoardView(board)
                own_left = set(view.words_by_role(Role.OWN, unrevealed_only=True))
                meant = reclue_rate.targets(sm, view, t["clue"], int(t["number"]))
                pending = {w for ws in sent for w in ws} & own_left
                out.append({"game": game_id, "turn": i, "team": us, "clue": t["clue"], "number": int(t["number"]),
                            "words": [w for w in board.words if w not in board.revealed],
                            "roles": {w: view.role_of(w).value for w in board.words if w not in board.revealed},
                            "past": [{k: p[k] for k in ("team", "clue", "number", "guesses")} for p in turns[:i]],
                            "targets": meant, "pending": sorted(pending), "own_left": len(own_left),
                            "opp_left": len(view.words_by_role(Role.OPPONENT, unrevealed_only=True))})
                sent.append(meant)
            for word, _ in t["guesses"]:
                board.reveal(word)
    return out


def sample(all_pos: list[dict], n_main: int, n_desperate: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(0)
    main = [p for p in all_pos if p["pending"] and not set(p["targets"]) & set(p["pending"])]
    rng.shuffle(main)
    seen, picked = set(), []
    for p in main:
        if p["game"] not in seen:
            seen.add(p["game"])
            picked.append(p)
    desperate = [p for p in all_pos if p["opp_left"] == 1 and p["own_left"] >= 3]
    rng.shuffle(desperate)
    return picked[:n_main], desperate[:n_desperate]


class Store:
    def __init__(self, db: Path):
        self.conn = sqlite3.connect(str(db), check_same_thread=False, timeout=60)
        self.lock = threading.Lock()
        self.conn.execute("""CREATE TABLE IF NOT EXISTS turn_probe (
            cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
            prompt TEXT NOT NULL, raw TEXT NOT NULL, guesses TEXT NOT NULL)""")
        self.conn.commit()

    def key(self, model: str, prompt: str) -> str:
        return json.dumps([model, VERSION, prompt])

    def get(self, model: str, prompt: str):
        with self.lock:
            row = self.conn.execute("SELECT guesses FROM turn_probe WHERE cache_key = ?",
                                    (self.key(model, prompt),)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, model: str, prompt: str, raw: str, guesses: list[str]) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO turn_probe VALUES (?, ?, ?, ?, ?, ?)",
                              (self.key(model, prompt), model, VERSION, prompt, raw, json.dumps(guesses)))
            self.conn.commit()


def parse(text: str, words: list[str], most: int) -> list[str]:
    """The last JSON array of strings in the answer, mapped to board words,
    without repeats, at most `most`. Raises ValueError if there is none or it
    names a word not on the board."""
    index = {w.lower(): w for w in words}
    for m in reversed(list(re.finditer(r"\[[^\[\]]*\]", text, re.S))):
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
            continue
        out = []
        for x in arr:
            w = index.get(x.strip().lower())
            if w is None:
                raise ValueError(f"{x!r} is not on the board")
            if w not in out:
                out.append(w)
        return out[:most]
    raise ValueError("no JSON array of words")


def ask(guesser, store: Store, prompt: str, words: list[str], most: int, usage: dict) -> list[str] | None:
    got = store.get(guesser.cache_model_id, prompt)
    if got is not None:
        return got
    problems = []
    for attempt in range(ATTEMPTS):
        request = {"model": guesser.model, "max_completion_tokens": guesser.max_tokens,
                   "messages": [{"role": "user", "content": prompt}], "reasoning_effort": guesser.reasoning_effort}
        resp = guesser._create(request)
        text = resp.choices[0].message.content or ""
        with store.lock:
            usage["calls"] += 1
            if resp.usage is not None:
                usage["in"] += resp.usage.prompt_tokens or 0
                usage["out"] += resp.usage.completion_tokens or 0
        try:
            guesses = parse(text, words, most)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        store.put(guesser.cache_model_id, prompt, text, guesses)
        return guesses
    usage["failed"] += 1
    print(f"  no usable answer: {problems}", flush=True)
    return None


def play(guesses: list[str], roles: dict) -> tuple[list[str], str]:
    """The guesses actually made: up to and including the first that is not ours."""
    made = []
    for w in guesses:
        made.append(w)
        if roles[w] != "own":
            return made, roles[w]
    return made, "stopped"


def report(name: str, pos: list[dict], arms: tuple[str, ...], answers: dict) -> None:
    print(f"\n{name}: {len(pos)} positions")
    print(f"  {'arm':14s} {'answered':>8s} {'listed':>7s} {'pending listed':>15s} {'pending found':>14s} "
          f"{'own found':>10s} {'wrong':>6s} {'assassin':>9s}")
    for arm in arms:
        rows = [(p, answers[(id(p), arm)]) for p in pos if answers.get((id(p), arm)) is not None]
        if not rows:
            continue
        n = len(rows)
        listed = sum(len(g) for _, g in rows) / n
        pend_any = sum(bool(set(g) & set(p["pending"])) for p, g in rows) / n
        found = [play(g, p["roles"]) for p, g in rows]
        pend_found = sum(bool(set(m) & set(p["pending"])) for (p, _), (m, _) in zip(rows, found)) / n
        own = sum(sum(p["roles"][w] == "own" for w in m) for (p, _), (m, _) in zip(rows, found)) / n
        wrong = sum(end != "stopped" for _, end in found) / n
        assassin = sum(end == "assassin" for _, end in found)
        print(f"  {arm:14s} {n:8d} {listed:7.2f} {pend_any:15.1%} {pend_found:14.1%} {own:10.2f} "
              f"{wrong:6.1%} {assassin:9d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--n-main", type=int, default=300)
    ap.add_argument("--n-desperate", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    main_pos, desp_pos = sample(positions(args.db), args.n_main, args.n_desperate)
    jobs = [(p, a) for p in main_pos for a in ("none", "history", "history_score")] + \
           [(p, a) for p in desp_pos for a in ("history", "history_score")]
    print(f"main {len(main_pos)}, desperate {len(desp_pos)}: {len(jobs)} prompts")
    if args.dry_run:
        p = main_pos[0]
        print(f"\nexample (game {p['game']}, turn {p['turn']}; targets {p['targets']}, pending {p['pending']}):\n")
        print(prompt_for(p, "history_score"))
        return

    from codenames.guessers.openai_compat import OpenAICompatGuesser

    guesser = OpenAICompatGuesser(MODEL, reasoning_effort="low")
    store = Store(args.db)
    usage = {"calls": 0, "in": 0, "out": 0, "failed": 0}
    answers = {}

    def run(job):
        p, arm = job
        answers[(id(p), arm)] = ask(guesser, store, prompt_for(p, arm), p["words"], p["number"] + 1, usage)

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(run, jobs), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    cost = usage["in"] * 0.037e-6 + usage["out"] * 0.17e-6
    print(f"calls {usage['calls']} (failed positions {usage['failed']}), tokens in {usage['in']} out {usage['out']}, "
          f"~${cost:.3f}")
    report("main (pending words the clue does not target)", main_pos, ("none", "history", "history_score"), answers)
    report("desperate (opponent on its last word, we have 3+)", desp_pos, ("history", "history_score"), answers)


if __name__ == "__main__":
    main()
