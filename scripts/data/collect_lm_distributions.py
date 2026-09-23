"""Score positions with a local language model: every candidate's probability
at every step, instead of one sampled ranking.

The positions are ones an API guesser already ranked (cache/llm_store.db):
same board, clue, candidate order and prompt. Our own sampler chose those
positions; the API guesser's answer is used only when asked for.

  --source own        (default) step j conditions on the local model's OWN
                      argmax picks: the local model is the whole teacher, and
                      these train a listener.
  --source teacher    step j conditions on the API guesser's actual pick order:
                      the local model scored as a predictor of that guesser,
                      for evaluation only.

Free (local GPU), resumable, written to cache/lm_distributions.db and never
to the paid-response store. Two backends:

    .venv/bin/python      scripts/data/collect_lm_distributions.py --backend hf
    .venv-vllm/bin/python scripts/data/collect_lm_distributions.py --backend vllm

    ... --teacher claude-sonnet-5+effort=medium --collected 0 --source teacher
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from codenames.listener_training import DEFAULT_MODEL, board_lookup, resolve_seed
from codenames.llm_store import DEFAULT_DB_PATH
from codenames.local_lm import DEFAULT_DIST_DB, DEFAULT_LM, DistributionStore, score_positions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default=DEFAULT_MODEL,
                    help="whose cached rankings define the positions")
    ap.add_argument("--source", choices=("own", "teacher"), default="own")
    ap.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--max-seed", type=int, default=60, help="arena seeds, as in train_listener.py")
    ap.add_argument("--collected", type=int, default=45000,
                    help="collected seeds; 45000 covers training (<40000) and the holdout range")
    ap.add_argument("--chunk", type=int, default=None,
                    help="positions per round-batch (default: 64 for hf, 2000 for vllm)")
    ap.add_argument("--max-rows", type=int, default=80, help="hf: candidates per forward pass")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_DIST_DB)
    args = ap.parse_args()
    source = "own" if args.source == "own" else args.teacher

    boards = board_lookup(args.max_seed, args.collected)
    con = sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True)
    rows = con.execute("SELECT clue, candidates, number, ranking FROM responses "
                       "WHERE model=? AND number IS NOT NULL", (args.teacher,)).fetchall()
    con.close()
    store = DistributionStore(args.out)
    done = store.done(args.lm, source)
    jobs, seen = [], set()
    for clue, cand_j, number, rank_j in rows:
        cand, rank = json.loads(cand_j), json.loads(rank_j)
        key = (clue, cand_j, int(number))
        if key in seen or key in done or sorted(cand) != sorted(rank) or resolve_seed(rank, boards) is None:
            continue
        seen.add(key)
        jobs.append((clue, cand, int(number), min(int(number), len(cand) - 1), rank))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(rows)} cached rankings from {args.teacher}; {len(jobs)} positions to score "
          f"(source {source}, backend {args.backend})", flush=True)
    if not jobs:
        return

    if args.backend == "hf":
        from codenames.local_lm import HFBackend
        backend, chunk = HFBackend(args.lm, max_rows=args.max_rows), args.chunk or 64
    else:
        from codenames.local_lm import VLLMBackend
        backend, chunk = VLLMBackend(args.lm), args.chunk or 2000

    t0 = time.time()
    for start in range(0, len(jobs), chunk):
        part = jobs[start:start + chunk]
        dists = score_positions(backend, part, own_path=args.source == "own")
        store.put_many(args.lm, source, [(j[0], j[1], j[2], d) for j, d in zip(part, dists)])
        n = start + len(part)
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(jobs)}  {rate:.1f}/s  eta {(len(jobs) - n) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
