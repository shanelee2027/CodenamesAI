"""Can gpt-oss be *prompted* to go back to earlier clues? A follow-up to
probe_history_guesser.py, whose turn-v1 prompt showed the game so far but
left out that a guess may go to an earlier clue's word: gpt-oss then listed a
pending word 3.3% of the time with or without the history (docs/log.md,
2026-10-27). Here the prompt says it, in the rulebook's terms: the extra guess
(number + 1) exists so that a team can go back to a clue it did not finish.

**Why measure it.** A human remembers the earlier clues. A spymaster that
knows this can leave a pending word alone and count on the spare guess,
instead of re-clueing it (the incumbent re-clues pending words on 74% of its
later clues). That only pays in the arena, and can only be learned from
rankings, if the guesser behaves that way. This asks whether a prompt can make
gpt-oss behave that way; it does not show that humans do.

**Positions.** The same 300 *main* positions as probe_history_guesser.py
(its `positions` and `sample`, seed 0): later incumbent clues in the comfort
arena with pending own words, none of which the current clue targets. A
pending word listed is therefore not explained by the current clue.

**Arms.**
- `history`: the 10-27 prompt (turn-v1) with the game so far. Its answers are
  already stored, so it costs nothing; the baseline.
- `rule`: the same, plus the rule sentence (RULE).
- `rule_nohist`: the rule without the history. The control: how often a
  pending word is listed when the guesser cannot know it is pending.
- `rule_medium`: `rule` at medium reasoning effort (the others are low).
- `rule_ask`: `rule`, and the answer first names, for each earlier clue of its
  spymaster, the board word it now thinks was meant (or null), then the
  guesses. The question that scripts/tools/probe_clue_history_turn.py asked in its
  structured arms, here together with the rule.
- `history_medium`, `rule_nohist_medium`: the controls for `rule_medium`,
  the history without the rule and the rule without the history, both at
  medium effort.
- `rule_chat`: the game as a conversation. Each earlier turn of its team is a
  user message with the clue and the board, then an assistant message with
  the guesses that were made, then the results and the other team's turns.
  The guesser "remembers" its own play instead of reading a summary.

Answers go to table `turn_probe` (prompt_version turn-v2-rule), keyed by the
prompt, so a re-run costs nothing.

    python scripts/data/probe_history_rule.py --dry-run   # positions and one prompt per arm, no calls
    python scripts/data/probe_history_rule.py             # ~1,500 calls, ~$0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_history_guesser as H  # noqa: E402

VERSION = "turn-v2-rule"
ARMS = ("history", "rule", "rule_nohist", "rule_medium", "rule_ask", "rule_chat", "history_medium",
        "rule_nohist_medium")

RULE = ("Your guesses do not have to be for the current clue. Any of your team's words still on the "
        "board counts, including words your spymaster meant for an earlier clue that your team did not "
        "find. That is what the extra guess is for: you may guess one more word than the clue's number "
        "so that you can go back to an earlier clue.")

ASK = ("First, for each earlier clue from your spymaster, say which word still on the board you now think "
       "it was meant for, or null if you think none is left. Then list the words you would guess, in the "
       "order you would guess them, and stop where you would stop. Respond with ONLY a JSON object, e.g. "
       '{"earlier": {"clue1": "word", "clue2": null}, "guesses": ["word1", "word2"]}, with no other text. '
       "The guesses may be empty.")

FORMAT = ("List the words you would guess, in the order you would guess them, and stop where you would "
          'stop. Respond with ONLY a JSON array of words, e.g. ["word1", "word2"], with no other text. '
          "The array may be empty.")


def prompt_for(pos: dict, arm: str) -> str:
    base = H.prompt_for(pos, "none" if arm.startswith("rule_nohist") else "history")
    if arm.startswith("history"):
        return base
    head, tail = base.split("\n\nList the words you would guess", 1)
    return head + " " + RULE + "\n\n" + (ASK if arm == "rule_ask" else FORMAT)


def chat_for(pos: dict) -> list[dict]:
    """The game so far as a conversation: our team's earlier turns are the
    guesser's own answers, everything else arrives in user messages."""
    us = pos["team"]
    intro = ("You are playing the guesser role in the board game Codenames. Your spymaster gives you a clue "
             "and a number each turn. You may make up to the number + 1 guesses, one at a time. A guess that "
             "is not one of your team's words ends the turn, and you may stop guessing at any point. " + RULE
             + " After each of your turns you are told what your guesses turned out to be and what the "
             "other team did.\n\n" + FORMAT)
    messages, pending_text = [], [intro]
    for i, t in enumerate(pos["past"]):
        mine = t["team"] == us
        if not mine:
            got = ", ".join(f"{w} ({H.SEEN[(r, False)]})" for w, r in t["guesses"])
            pending_text.append(f'The other team\'s spymaster gave "{t["clue"]}" for {t["number"]}. '
                                + (f"They guessed {got}." if got else "They guessed nothing."))
            continue
        gone = set(w for t2 in pos["past"][:i] for w, _ in t2["guesses"])
        on_board = [w for w in pos["all_words"] if w not in gone]
        pending_text.append(f'Your spymaster gives the clue "{t["clue"]}" for {t["number"]} word(s). Here are '
                            f"the words still on the board:\n\n" + "\n".join(on_board))
        messages.append({"role": "user", "content": "\n\n".join(pending_text)})
        messages.append({"role": "assistant", "content": json.dumps([w for w, _ in t["guesses"]])})
        outcome = "; ".join(f"{w}: {H.SEEN[(r, True)]}" for w, r in t["guesses"]) or "you guessed nothing"
        pending_text = [f"Result: {outcome}."]
    pending_text.append(f'Your spymaster now gives the clue "{pos["clue"]}" for {pos["number"]} word(s). Here are '
                        f"the words still on the board:\n\n" + "\n".join(pos["words"]))
    messages.append({"role": "user", "content": "\n\n".join(pending_text)})
    return messages


def parse_ask(text: str, words: list[str], most: int) -> tuple[list[str], dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"bad JSON: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("guesses"), list):
        raise ValueError("no guesses list")
    guesses = H.parse(json.dumps(obj["guesses"]), words, most)
    earlier = obj.get("earlier") if isinstance(obj.get("earlier"), dict) else {}
    return guesses, earlier


def ask(guesser, store: H.Store, key_prompt: str, messages: list[dict], words: list[str], most: int,
        usage: dict, ask_form: bool) -> tuple[list[str], str] | None:
    """(guesses, raw answer), from the store when the same prompt was asked."""
    with store.lock:
        row = store.conn.execute("SELECT guesses, raw FROM turn_probe WHERE cache_key = ?",
                                 (store.key(guesser.cache_model_id, key_prompt),)).fetchone()
    if row is not None:
        return json.loads(row[0]), row[1]
    problems = []
    for _ in range(H.ATTEMPTS):
        request = {"model": guesser.model, "max_completion_tokens": guesser.max_tokens,
                   "messages": messages, "reasoning_effort": guesser.reasoning_effort}
        resp = guesser._create(request)
        text = resp.choices[0].message.content or ""
        with store.lock:
            usage["calls"] += 1
            usage["length_stops"] += resp.choices[0].finish_reason == "length"
            if resp.usage is not None:
                usage["in"] += resp.usage.prompt_tokens or 0
                usage["out"] += resp.usage.completion_tokens or 0
        try:
            guesses = parse_ask(text, words, most)[0] if ask_form else H.parse(text, words, most)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        store.put(guesser.cache_model_id, key_prompt, text, guesses)
        return guesses, text
    with store.lock:
        usage["failed"] += 1
    print(f"  no usable answer: {problems}", flush=True)
    return None


def report(pos: list[dict], answers: dict) -> None:
    print(f"\nmain (pending words the clue does not target): {len(pos)} positions, "
          f"number {sum(p['number'] for p in pos) / len(pos):.2f}, pending {sum(len(p['pending']) for p in pos) / len(pos):.2f}")
    print(f"  {'arm':12s} {'answered':>8s} {'listed':>7s} {'used +1':>8s} {'pending listed':>15s} "
          f"{'pending found':>14s} {'own found':>10s} {'wrong':>6s} {'assassin':>9s} {'names pending':>14s}")
    for arm in ARMS:
        rows = [(p, answers[(id(p), arm)]) for p in pos if answers.get((id(p), arm)) is not None]
        if not rows:
            continue
        n = len(rows)
        listed = sum(len(g) for _, (g, _) in rows) / n
        plus1 = sum(len(g) > p["number"] for p, (g, _) in rows) / n
        pend_any = sum(bool(set(g) & set(p["pending"])) for p, (g, _) in rows) / n
        made = [H.play(g, p["roles"]) for p, (g, _) in rows]
        pend_found = sum(bool(set(m) & set(p["pending"])) for (p, _), (m, _) in zip(rows, made)) / n
        own = sum(sum(p["roles"][w] == "own" for w in m) for (p, _), (m, _) in zip(rows, made)) / n
        wrong = sum(end != "stopped" for _, end in made) / n
        assassin = sum(end == "assassin" for _, end in made)
        names = ""
        if arm == "rule_ask":
            hit = 0
            for p, (g, raw) in rows:
                try:
                    earlier = parse_ask(raw, p["words"], p["number"] + 1)[1]
                except ValueError:
                    earlier = {}
                said = {str(v).lower() for v in earlier.values() if v}
                hit += bool(said & {w.lower() for w in p["pending"]})
            names = f"{hit / n:.1%}"
        print(f"  {arm:12s} {n:8d} {listed:7.2f} {plus1:8.1%} {pend_any:15.1%} {pend_found:14.1%} {own:10.2f} "
              f"{wrong:6.1%} {assassin:9d} {names:>14s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pos, _ = H.sample(H.positions(args.db), args.n, 0)
    jobs = [(p, a) for p in pos for a in ARMS]
    print(f"{len(pos)} positions, {len(jobs)} prompts")
    if args.dry_run:
        p = max(pos, key=lambda q: len(q["past"]))
        print(f"\nexample (game {p['game']}, turn {p['turn']}; targets {p['targets']}, pending {p['pending']})")
        for arm in ARMS[1:]:
            print(f"\n===== {arm}\n")
            if arm == "rule_chat":
                for m in chat_for(p):
                    print(f"--- {m['role']}\n{m['content']}\n")
            else:
                print(prompt_for(p, arm))
        return

    from codenames.guessers.openai_compat import OpenAICompatGuesser

    low = OpenAICompatGuesser(H.MODEL, reasoning_effort="low")
    medium = OpenAICompatGuesser(H.MODEL, reasoning_effort="medium")
    old, new = H.Store(args.db, H.VERSION), H.Store(args.db, VERSION)
    usage = {"calls": 0, "in": 0, "out": 0, "failed": 0, "length_stops": 0}
    answers = {}

    def run(job):
        p, arm = job
        most = p["number"] + 1
        if arm == "rule_chat":
            msgs = chat_for(p)
            key = json.dumps(msgs)
        else:
            key = prompt_for(p, arm)
            msgs = [{"role": "user", "content": key}]
        answers[(id(p), arm)] = ask(medium if arm.endswith("medium") else low, old if arm == "history" else new,
                                    key, msgs, p["words"], most, usage, arm == "rule_ask")

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(run, jobs), 1):
            if i % 300 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    cost = usage["in"] * 0.037e-6 + usage["out"] * 0.17e-6
    print(f"calls {usage['calls']} (failed positions {usage['failed']}, {usage['length_stops']} hit max tokens), tokens in {usage['in']} out {usage['out']}, "
          f"~${cost:.3f}")
    report(pos, answers)


if __name__ == "__main__":
    main()
