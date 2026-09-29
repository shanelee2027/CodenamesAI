"""The incumbent's picks and values on simulated positions: the imitation
policy's training data (codenames/clue_policy.py, docs/log.md
"gptoss_reward_policy: design", choice 4).

    python scripts/data/collect_imitation_data.py --split train --n 40000
    python scripts/data/collect_imitation_data.py --split val --n 2000

Per position: the board (word ids, role ids from the side to move,
unrevealed flags), the incumbent's clue and number, and its expected reward
for every k of each of its 200 shortlisted clues. Free: no LLM is called.
Positions come from `clue_policy.positions` over the split's seed range, so
train and val never share a board.

The incumbent plays uncapped (max_number=None): it may announce every own
word left, and the net values run to k = POLICY_MAX_NUMBER.

Writes cache/training_data/policy_imitation_<split>_uncapped.npz, from shards
of 2,000 positions kept in policy_imitation_<split>_uncapped_shards/ so a
rerun resumes. (The capped-at-4 data the first imitation_policy was trained
on is policy_imitation_<split>.npz.)
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from codenames.clue_policy import (
    IMITATION_SEEDS,
    POLICY_MAX_NUMBER,
    VAL_SEEDS,
    PolicyFeatures,
    incumbent_targets,
    positions,
)
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

OUT_DIR = DEFAULT_CACHE_DIR / "training_data"
SHORTLIST = 200
SHARD = 2000
_state: dict = {}


def _init() -> None:
    import torch

    from codenames.spymasters.learned_listener import LearnedListenerSpymaster

    # One thread per worker for torch (expected_words' ndtr) and LightGBM: with
    # every worker spawning a full thread pool, 14 workers ran at a third of
    # their single-process rate.
    torch.set_num_threads(1)

    _state["sims"] = SimilarityTensor.load()
    # The incumbent uncapped, so the policy learns from numbers up to every
    # own word left.
    _state["sm"] = LearnedListenerSpymaster(max_number=None)
    feats = PolicyFeatures.load()
    _state["feats"] = feats
    _state["pool_pos"] = {int(c): i for i, c in enumerate(feats.pool)}


def _one(seed: int) -> dict:
    from codenames.clue_policy import deal_position

    view = deal_position(seed)
    t = incumbent_targets(_state["sm"], view, _state["sims"])
    feats, pos, sims = _state["feats"], _state["pool_pos"], _state["sims"]
    b = feats.encode(view)
    short = np.full(SHORTLIST, -1, dtype=np.int32)
    net = np.full((SHORTLIST, POLICY_MAX_NUMBER), np.nan, dtype=np.float32)
    idx = [pos[int(c)] for c in t["shortlist"]]          # the incumbent's pool is this pool
    short[: len(idx)] = idx
    net[: len(idx), : t["net"].shape[1]] = t["net"]
    return {"seed": seed, "words": b.words, "roles": b.roles, "present": b.present,
            "pick": pos[sims.clue_index[t["clue"].lower()]], "number": t["number"],
            "score": t["score"], "shortlist": short, "net": net}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    seeds = [s for s, _ in positions(IMITATION_SEEDS if args.split == "train" else VAL_SEEDS, args.n)]
    # Shards of SHARD positions, each written as soon as it is done and skipped
    # on a rerun: a run killed at 26k of 30k (the terminal closed) had kept
    # everything in memory and lost it all.
    shard_dir = OUT_DIR / f"policy_imitation_{args.split}_uncapped_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [seeds[i: i + SHARD] for i in range(0, len(seeds), SHARD)]
    todo = [(j, s) for j, s in enumerate(shards) if not (shard_dir / f"{s[0]}_{len(s)}.npz").exists()]
    print(f"{len(shards)} shards, {len(shards) - len(todo)} already on disk", flush=True)
    t0, done = time.time(), 0
    with Pool(args.workers, initializer=_init) as p:
        for j, shard in todo:
            rows = p.map(_one, shard, chunksize=8)
            np.savez(shard_dir / f"{shard[0]}_{len(shard)}.npz",
                     **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            done += len(shard)
            print(f"  shard {j + 1}/{len(shards)}  {done / (time.time() - t0):.1f} positions/s", flush=True)

    parts = [np.load(shard_dir / f"{s[0]}_{len(s)}.npz") for s in shards]
    out = OUT_DIR / f"policy_imitation_{args.split}_uncapped.npz"
    np.savez(out, **{k: np.concatenate([d[k] for d in parts]) for k in parts[0].files})
    numbers = np.bincount(np.concatenate([d["number"] for d in parts]), minlength=POLICY_MAX_NUMBER + 1)[1:]
    print(f"wrote {out}: {len(seeds)} positions in {time.time() - t0:.0f}s; incumbent numbers 1-{POLICY_MAX_NUMBER}: {numbers.tolist()}")


if __name__ == "__main__":
    main()
