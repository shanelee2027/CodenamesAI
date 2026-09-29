"""Which cheap LLM guesser agrees best with Sonnet? A screen for a second
evaluation guesser beside gpt-oss (docs/log.md, "Screening a second cheap
guesser").

Positions are the ones Sonnet already ranked on the frozen suite's boards
(holdout_v1: held-out words only), with the exact clue, candidate list and
number Sonnet was given, so each candidate sees Sonnet's prompt. A fixed random
sample of them is sent to every guesser spec, through the project's own
guesser code (build_guesser), so answers are cached in cache/llm_store.db and
parsed exactly as in games. A position already cached for a spec costs
nothing.

Reported per guesser, against Sonnet's ranking of the same position:
- **first pick agrees**: the guesser's top word is Sonnet's top word.
- **top-k overlap**: |guesser top k & Sonnet top k| / k, with k the clue's
  number: the words a turn would actually read.
- **the turn agrees**: the same k words, in any order.
- **Kendall tau** over the full rankings.
- **refused**: positions the guesser would not rank after its retries.
- **cost**: measured tokens at DeepInfra's listed prices (PRICES below).

    python scripts/tools/screen_guessers.py deepinfra:openai/gpt-oss-120b:low \\
        deepinfra:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B:low --n 500
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from train_listener_net import load_sets  # noqa: E402

from codenames.env import load_env  # noqa: E402
from codenames.guessers.registry import build_guesser  # noqa: E402

# USD per million tokens (input, output), DeepInfra's list prices on 2026-09-29.
PRICES = {
    "openai/gpt-oss-120b": (0.037, 0.17),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": (0.085, 0.40),
    "zai-org/GLM-4.5-Air": (0.20, 1.10),
}


def kendall(a: list[str], b: list[str]) -> float:
    from scipy.stats import kendalltau

    common = [w for w in a if w in set(b)]
    if len(common) < 3:
        return float("nan")
    pos = {w: i for i, w in enumerate(b)}
    return float(kendalltau(range(len(common)), [pos[w] for w in common]).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=48)
    args = ap.parse_args()
    load_env()

    son = load_sets()["held-out words, Sonnet"]
    rng = np.random.default_rng(0)
    sample = [son[i] for i in sorted(rng.choice(len(son), size=min(args.n, len(son)), replace=False))]
    # Sonnet's prompt (the candidates in the order it saw them) and its ranking.
    items = [(p["key"][0], list(p["key"][1]), p["key"][2], [p["words"][t] for t in p["targets"]])
             for p in sample]
    print(f"{len(items)} Sonnet positions on held-out-word boards")

    for spec in args.specs:
        g = build_guesser(spec)
        usage = {"calls": 0, "in": 0, "out": 0}
        create = g.client.chat.completions.create

        def metered(*a, **k):
            # Smaller hosted models answer 429 "model busy" under load: back
            # off and retry, rather than let one busy moment count as a refusal.
            import openai

            for wait in (2, 4, 8, 16, 32, 64, None):
                try:
                    r = create(*a, **k)
                    break
                except openai.RateLimitError:
                    if wait is None:
                        raise
                    usage["busy"] = usage.get("busy", 0) + 1
                    time.sleep(wait * (1 + np.random.rand()))
            usage["calls"] += 1
            if r.usage:
                usage["in"] += r.usage.prompt_tokens
                usage["out"] += r.usage.completion_tokens
            return r

        g.client.chat.completions.create = metered

        def one(it):
            clue, cand, number, _ = it
            try:
                return g.rank_candidates(clue, cand, None, number=number)
            except RuntimeError:
                return None

        t0 = time.time()
        with ThreadPoolExecutor(args.threads) as ex:
            got = list(ex.map(one, items))
        first, overlap, turn, tau = [], [], [], []
        for (_, _, k, sr), r in zip(items, got):
            if r is None:
                continue
            first.append(r[0] == sr[0])
            overlap.append(len(set(r[:k]) & set(sr[:k])) / k)
            turn.append(set(r[:k]) == set(sr[:k]))
            tau.append(kendall(r, sr))
        refused = sum(r is None for r in got)
        pin, pout = PRICES.get(g.model, (float("nan"), float("nan")))
        dollars = (usage["in"] * pin + usage["out"] * pout) / 1e6
        print(f"\n{spec}")
        print(f"  first pick agrees {np.mean(first):.3f}   top-k overlap {np.mean(overlap):.3f}   "
              f"turn agrees {np.mean(turn):.3f}   Kendall tau {np.nanmean(tau):.3f}")
        print(f"  refused {refused}/{len(items)}   paid requests {usage['calls']} "
              f"(mean {usage['out'] / max(usage['calls'], 1):.0f} completion tokens)   "
              f"${dollars:.3f} (${1000 * dollars / max(usage['calls'], 1):.3f} per 1k requests)   "
              f"{time.time() - t0:.0f}s   busy retries {usage.get('busy', 0)}")


if __name__ == "__main__":
    main()
