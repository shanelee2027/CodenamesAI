"""Turns from the history-aware gpt-oss guesser, for fitting how often it goes
back to an earlier clue (docs/log.md, "Clue-history probe 3").

**The guesser** is probe_history_rule.py's `rule_medium` arm: the turn-level
prompt with the game so far and the rulebook's rule for the extra guess, at
medium reasoning effort. Of the prompts tried it is the only one that makes
gpt-oss list a pending word (an own word an earlier clue was meant for, still
unfound) much more often than chance (32% against 3.3%).

**Positions.** Incumbent clues in the comfort arena's recorded games
(probe_history_guesser.positions) with at least one pending word, one per
game, from games the probe did not use, so that the probe's 300 stay a test
set; a game's main position where it has one. Both kinds are kept and labelled: `main`, the current clue targets none
of the pending words, and `reclue`, it targets at least one. A model of when
the guesser picks a pending word up needs both: the first is what a spymaster
that stops re-clueing would face, the second what the incumbent does now.

Answers go to table `turn_probe` (prompt_version turn-v2-rule, as the probe),
so a re-run costs nothing. The positions with their answers are written to
`--out` for fitting.

    python scripts/data/collect_history_turns.py --dry-run     # counts, no calls
    python scripts/data/collect_history_turns.py --n 2000      # ~$0.60
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_history_guesser as H  # noqa: E402
import probe_history_rule as R  # noqa: E402

ARM = "rule_medium"
MAX_TOKENS = 8192            # 58 of 560 medium calls in the probe hit 4,096


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "cache" / "training_data" / "history_turns.pkl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    every = H.positions(args.db)
    probe, _ = H.sample(every, 300, 0)
    held = {p["game"] for p in probe}
    rng = random.Random(1)
    cands = [p for p in every if p["pending"] and p["game"] not in held]
    rng.shuffle(cands)
    # A game's main position if it has one: the incumbent re-clues pending
    # words on most of its clues, so main positions are the scarce kind.
    for p in cands:
        p["kind"] = "reclue" if set(p["targets"]) & set(p["pending"]) else "main"
    cands.sort(key=lambda p: p["kind"] != "main")
    seen, pos = set(), []
    for p in cands:
        if p["game"] not in seen:
            seen.add(p["game"])
            pos.append(p)
    rng.shuffle(pos)
    pos = pos[:args.n]
    kinds = {k: sum(p["kind"] == k for p in pos) for k in ("main", "reclue")}
    print(f"{len(every)} incumbent clues, {len(cands)} with pending words outside the probe's games; "
          f"{len(pos)} positions (one per game): {kinds}")
    if args.dry_run:
        return

    from codenames.guessers.openai_compat import OpenAICompatGuesser

    guesser = OpenAICompatGuesser(H.MODEL, reasoning_effort="medium", max_tokens=MAX_TOKENS)
    store = H.Store(args.db, R.VERSION)
    usage = {"calls": 0, "in": 0, "out": 0, "failed": 0, "length_stops": 0}

    def run(p):
        prompt = R.prompt_for(p, ARM)
        got = R.ask(guesser, store, prompt, [{"role": "user", "content": prompt}], p["words"], p["number"] + 1,
                    usage, False)
        p["guesses"] = None if got is None else got[0]

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(run, pos), 1):
            if i % 250 == 0:
                print(f"  {i}/{len(pos)}", flush=True)
    cost = usage["in"] * 0.037e-6 + usage["out"] * 0.17e-6
    print(f"calls {usage['calls']} (failed positions {usage['failed']}, {usage['length_stops']} hit max tokens), "
          f"tokens in {usage['in']} out {usage['out']}, ~${cost:.3f}")
    done = [p for p in pos if p["guesses"] is not None]
    args.out.write_bytes(pickle.dumps(done, protocol=pickle.HIGHEST_PROTOCOL))
    for k in ("main", "reclue"):
        ps = [p for p in done if p["kind"] == k]
        if ps:
            listed = sum(bool(set(p["guesses"]) & set(p["pending"]) - set(p["targets"])) for p in ps) / len(ps)
            print(f"  {k:7s} {len(ps):5d}: an untargeted pending word listed {listed:.1%}")
    print(f"saved {len(done)} -> {args.out}")


if __name__ == "__main__":
    main()
