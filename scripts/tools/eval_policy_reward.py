"""Mean per-turn reward on validation positions through one guesser, for
several spymasters side by side: the RL stage's yardstick
(docs/versions/gptoss_reward_policy.md).

    python scripts/tools/eval_policy_reward.py learned_listener imitation_policy gptoss_reward_policy --n 500

Every spymaster gives its clue and number on the same positions
(clue_policy.VAL_SEEDS: training vocabulary, seeds no stage trained on). The
guesser ranks the unrevealed words with no number announced, and the turn is
scored at the number the spymaster announced (clue_policy.rollout), under the
arena's rules. The same (position, clue) is one cached call, so two
spymasters that give the same clue get the identical sample and their
difference on that position is exactly zero: the comparison is paired.

Reported per spymaster: mean reward and its SE, own words per turn, how turns
ended, mean k; and the paired difference from the first spymaster named, with
a 95% interval. A `name=path` argument evaluates a checkpoint in place of the
registered one (`gptoss_reward_policy=cache/some.pt`). Rows go to
cache/training_data/policy_val_eval.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from codenames.clue_policy import CAP, MISS_ROLES, VAL_SEEDS, positions, rollout
from codenames.eval_suite import spymaster_identity
from codenames.guessers.registry import build_guesser
from codenames.similarity import DEFAULT_CACHE_DIR

GUESSER = "deepinfra:openai/gpt-oss-120b:low:temperature=1.0"
OUT = DEFAULT_CACHE_DIR / "training_data" / "policy_val_eval.jsonl"
_state: dict = {}


def _spec(arg: str):
    from codenames.spymasters.registry import spymaster_spec

    name, _, path = arg.partition("=")
    cls, kwargs = spymaster_spec(name, **({"model_path": path} if path else {}))
    return name, cls, kwargs


def _init(arg: str) -> None:
    import torch

    from codenames.similarity import SimilarityTensor

    torch.set_num_threads(1)
    _, cls, kwargs = _spec(arg)
    _state["sm"] = cls(**kwargs)
    _state["sims"] = SimilarityTensor.load()


def _pick(n_and_seed: tuple[int, int]) -> tuple[int, str, int]:
    from codenames.clue_policy import deal_position
    from codenames.spymasters.base import TurnContext

    i, seed = n_and_seed
    clue, number, _ = _state["sm"].top_clues(TurnContext(deal_position(seed), 0), _state["sims"], 1)[0]
    return i, clue, number


def picks_for(arg: str, seeds: list[int], workers: int) -> list[tuple[str, int]]:
    out = [None] * len(seeds)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(arg,)) as ex:
        for i, clue, number in ex.map(_pick, list(enumerate(seeds)), chunksize=4):
            out[i] = (clue, number)
    return out


def summarise(rows: list[dict]) -> dict:
    r = np.array([x["reward"] for x in rows])
    own, ends = [], {"all": 0, **{role.value: 0 for role in MISS_ROLES}}
    for x in rows:
        j, e = divmod(x["outcome"], len(MISS_ROLES)) if x["outcome"] != CAP else (x["number"], None)
        if e is None or j >= x["number"]:
            own.append(x["number"])
            ends["all"] += 1
        else:
            own.append(j)
            ends[MISS_ROLES[e].value] += 1
    n = len(rows)
    return {"n": n, "reward": r.mean(), "se": r.std(ddof=1) / math.sqrt(n), "own": float(np.mean(own)),
            "mean_k": float(np.mean([x["number"] for x in rows])),
            **{f"end_{k}": v / n for k, v in ends.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spymasters", nargs="+")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--guesser", default=GUESSER)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    views = positions(VAL_SEEDS, args.n)
    guesser = build_guesser(args.guesser)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[int, dict]] = {}
    with OUT.open("a") as log, ThreadPoolExecutor(args.threads) as pool:
        for arg in args.spymasters:
            t0 = time.time()
            name, cls, kwargs = _spec(arg)
            ident = spymaster_identity(name, kwargs, cls.model_files(kwargs))
            picks = picks_for(arg, [s for s, _ in views], args.workers)

            def one(i):
                try:
                    return rollout(guesser, views[i][1], picks[i][0])
                except Exception as exc:                                  # noqa: BLE001
                    return exc
            got = list(pool.map(one, range(len(views))))
            rows = {}
            for (seed, _), (clue, number), g in zip(views, picks, got):
                if isinstance(g, Exception):
                    continue
                row = {"spymaster": ident, "arg": arg, "guesser": args.guesser, "seed": seed, "clue": clue,
                       "number": number, "reward": g["rewards"][number - 1], "outcome": g["outcome"],
                       "rewards": g["rewards"]}
                rows[seed] = row
                log.write(json.dumps(row) + "\n")
            results[arg] = rows
            print(f"{arg}: {len(rows)}/{len(views)} scored in {time.time() - t0:.0f}s", flush=True)

    first = args.spymasters[0]
    common = sorted(set.intersection(*(set(r) for r in results.values())))
    print(f"\n{len(common)} positions scored for every spymaster, guesser {args.guesser}\n")
    print(f"{'spymaster':45s} {'reward':>7s} {'se':>6s} {'own':>5s} {'k':>5s} {'all N':>6s} "
          f"{'neutral':>8s} {'opp.':>6s} {'assassin':>9s}   diff vs {first} [95% CI]")
    for arg, rows in results.items():
        s = summarise([rows[x] for x in common])
        d = np.array([rows[x]["reward"] - results[first][x]["reward"] for x in common])
        same = np.mean([rows[x]["clue"] == results[first][x]["clue"] for x in common])
        diff = "" if arg == first else (f"   {d.mean():+.3f} [{d.mean() - 1.96 * d.std(ddof=1) / math.sqrt(len(d)):+.3f}, "
                                         f"{d.mean() + 1.96 * d.std(ddof=1) / math.sqrt(len(d)):+.3f}]  same clue {same:.0%}")
        print(f"{arg:45s} {s['reward']:7.3f} {s['se']:6.3f} {s['own']:5.2f} {s['mean_k']:5.2f} {s['end_all']:6.1%} "
              f"{s['end_neutral']:8.1%} {s['end_opponent']:6.1%} {s['end_assassin']:9.1%}{diff}")


if __name__ == "__main__":
    main()
