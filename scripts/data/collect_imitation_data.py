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

Writes cache/training_data/policy_imitation_<split>.npz.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from codenames.board import MAX_CLUE_NUMBER
from codenames.clue_policy import (
    IMITATION_SEEDS,
    VAL_SEEDS,
    PolicyFeatures,
    incumbent_targets,
    positions,
)
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

OUT_DIR = DEFAULT_CACHE_DIR / "training_data"
SHORTLIST = 200
_state: dict = {}


def _init() -> None:
    import torch

    from codenames.spymasters.learned_listener import LearnedListenerSpymaster

    # One thread per worker for torch (expected_words' ndtr) and LightGBM: with
    # every worker spawning a full thread pool, 14 workers ran at a third of
    # their single-process rate.
    torch.set_num_threads(1)

    _state["sims"] = SimilarityTensor.load()
    _state["sm"] = LearnedListenerSpymaster()
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
    net = np.full((SHORTLIST, MAX_CLUE_NUMBER), np.nan, dtype=np.float32)
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
    t0 = time.time()
    rows = []
    with Pool(args.workers, initializer=_init) as p:
        for i, r in enumerate(p.imap(_one, seeds, chunksize=8), 1):
            rows.append(r)
            if i % 1000 == 0 or i == len(seeds):
                el = time.time() - t0
                print(f"  {i}/{len(seeds)}  {i / el:.1f} positions/s", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"policy_imitation_{args.split}.npz"
    np.savez(out, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    numbers = np.bincount([r["number"] for r in rows], minlength=MAX_CLUE_NUMBER + 1)[1:]
    print(f"wrote {out}: {len(rows)} positions in {time.time() - t0:.0f}s; incumbent numbers 1-4: {numbers.tolist()}")


if __name__ == "__main__":
    main()
