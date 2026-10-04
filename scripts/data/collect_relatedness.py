"""Buy graded relatedness labels (codenames/relatedness.py) for the listener's
existing positions.

Each position is one already ranked by gpt-oss (the training data): the same
clue and the same candidate list in the same order, so the stored ranking
can order the words within each label set. The number is not given.

    # pilot: 200 generated training positions, with 20 ranking calls for a token comparison
    python scripts/data/collect_relatedness.py --n 200 --measure-ranking 20 --examples 12

    # everything in train, val and new boards, never more than 30,000 API calls
    python scripts/data/collect_relatedness.py --source all --max-calls 30000

Spend is reported in calls and tokens. Dollars are estimated from the
measured ~$0.07 per 1,000 ranking calls (docs/log.md), scaled by this
prompt's tokens per call against the ranking prompt's when
--measure-ranking was run.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))

DB = PROJECT_ROOT / "cache" / "llm_store.db"
SEED_BASE = 1_000_000                       # collect_listener_data.py: generated boards start here
RANK_COST_PER_CALL = 0.07 / 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["train", "val", "new boards"])
    ap.add_argument("--source", choices=("generated", "all"), default="generated",
                    help="generated: only collect_listener_data.py's sampled positions; all: also the "
                         "positions from recorded games")
    ap.add_argument("--n", type=int, default=None, help="positions (a fixed random subset); default all")
    ap.add_argument("--max-calls", type=int, required=True, help="stop starting API calls after this many")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--measure-ranking", type=int, default=0,
                    help="also send the ranking prompt (uncached, not stored) on this many positions")
    ap.add_argument("--examples", type=int, default=0, help="print this many labelled positions")
    ap.add_argument("--version", default=None, help="prompt version (codenames/relatedness.py PROMPTS); default the latest")
    ap.add_argument("--compare", default=None, help="also show this version's labels in the examples")
    args = ap.parse_args()

    from codenames.guessers.llm import _PROMPT_TEMPLATE
    from codenames.guessers.openai_compat import OpenAICompatGuesser
    from codenames.relatedness import PROMPT_VERSION, RelatednessAsker, RelatednessStore
    from train_listener_net import load_sets

    sets = load_sets()
    pos = [p for s in args.sets for p in sets[s] if args.source == "all" or p["seed"] >= SEED_BASE]
    pos = [pos[i] for i in np.random.default_rng(0).permutation(len(pos))]
    if args.n is not None:
        pos = pos[: args.n]
    g = OpenAICompatGuesser(model="openai/gpt-oss-120b", provider="deepinfra", reasoning_effort="low")
    version = args.version or PROMPT_VERSION
    asker = RelatednessAsker(g, RelatednessStore(DB), version)
    todo = [p for p in pos if asker.store.get(asker.model, p["key"][0], list(p["key"][1]), version) is None]
    print(f"{len(pos)} positions ({args.source}, {', '.join(args.sets)}); {len(pos) - len(todo)} already labelled, "
          f"{len(todo)} to buy; at most {args.max_calls} calls", flush=True)

    lock = threading.Lock()
    state = {"done": 0, "errors": 0}
    t0 = time.time()

    def work(p):
        with lock:
            if asker.usage["calls"] >= args.max_calls:
                return
        try:
            asker.ask(p["key"][0], list(p["key"][1]))
        except Exception as exc:                      # one bad position must not stop the run
            with lock:
                state["errors"] += 1
                if state["errors"] <= 5:
                    print(f"  error on {p['key'][0]!r}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
        with lock:
            state["done"] += 1
            if state["done"] % 500 == 0:
                u = asker.usage
                print(f"  {state['done']}/{len(todo)}  {u['calls']} calls  {state['done'] / (time.time() - t0):.1f}/s  "
                      f"errors {state['errors']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(work, todo))
    u = asker.usage
    per_call = (u["input_tokens"] + u["output_tokens"]) / max(u["calls"], 1)
    print(f"\ncalls {u['calls']} (retries {u['retries']}), errors {state['errors']}, unknown names {u['unknown_names']}")
    print(f"tokens per call: input {u['input_tokens'] / max(u['calls'], 1):.0f}, "
          f"output {u['output_tokens'] / max(u['calls'], 1):.0f}")

    scale = 1.0
    if args.measure_ranking:
        rin = rout = 0
        for p in pos[: args.measure_ranking]:
            clue, cands, k = p["key"]
            prompt = _PROMPT_TEMPLATE.format(clue=clue, count_note=f" for {k} word(s)", words="\n".join(cands))
            r = g._create({"model": g.model, "max_completion_tokens": g.max_tokens, "reasoning_effort": "low",
                           "messages": [{"role": "user", "content": prompt}]})
            rin, rout = rin + r.usage.prompt_tokens, rout + r.usage.completion_tokens
        n = args.measure_ranking
        print(f"ranking prompt, {n} calls: input {rin / n:.0f}, output {rout / n:.0f} tokens per call")
        scale = per_call / ((rin + rout) / n)
    print(f"estimated spend: ${u['calls'] * RANK_COST_PER_CALL * scale:.3f}"
          + (f" (this prompt uses {scale:.2f}x the ranking prompt's tokens)" if args.measure_ranking else
             " (at the ranking prompt's cost per call)"))

    # What the labels look like.
    labelled = [(p, asker.store.get(asker.model, p["key"][0], list(p["key"][1]), version)) for p in pos]
    labelled = [(p, r) for p, r in labelled if r is not None]
    ng = np.array([len(r[0]) for _, r in labelled])
    ns = np.array([len(r[1]) for _, r in labelled])
    print(f"\n{len(labelled)} labelled positions ({version})")
    for name, x in (("guess", ng), ("stretch", ns), ("guess + stretch", ng + ns)):
        print(f"  {name:16s} mean {x.mean():.2f}  " + "  ".join(f"{v}: {(x == v).mean():.0%}" for v in range(6))
              + f"  6+: {(x >= 6).mean():.0%}")
    for p, (guess, stretch) in labelled[: args.examples]:
        ranked = [p["words"][t] for t in p["targets"][:6]]
        print(f"\n  {p['key'][0]!r} (ranking asked for {p['key'][2]})\n    guess:   {guess}\n    stretch: {stretch}")
        if args.compare:
            old = asker.store.get(asker.model, p["key"][0], list(p["key"][1]), args.compare)
            if old is not None:
                print(f"    {args.compare}: guess {old[0]}  stretch {old[1]}")
        print(f"    ranking: {ranked} ...")


if __name__ == "__main__":
    main()
