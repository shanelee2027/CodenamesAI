"""Score cached positions with a local language model: every candidate's
probability at every step, instead of one sampled ranking.

The positions are ones an API guesser already ranked (cache/llm_store.db),
asked with the same prompt and the same candidate order, and step j is
conditioned on the order THAT guesser actually picked in. So each position's
Plackett-Luce choice groups are exactly the groups the listener was trained
on, and the only thing that changes is the label: a full distribution instead
of the one word that was drawn. The same numbers score the local model as a
predictor of the API guesser's real picks.

Free (local GPU), resumable, and written to cache/lm_distributions.db, never
to the paid-response store.

    python scripts/data/collect_lm_distributions.py                 # the gpt-oss positions
    python scripts/data/collect_lm_distributions.py --teacher claude-sonnet-5+effort=medium
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time

from codenames.listener_training import DEFAULT_MODEL, board_lookup, resolve_seed
from codenames.llm_store import DEFAULT_DB_PATH
from codenames.local_lm import DEFAULT_DIST_DB, DEFAULT_LM, DistributionStore, LocalLMScorer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default=DEFAULT_MODEL,
                    help="whose cached rankings define the positions and the conditioning order")
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--max-seed", type=int, default=60, help="arena seeds, as in train_listener.py")
    ap.add_argument("--collected", type=int, default=45000,
                    help="collected seeds; 45000 covers training (<40000) and the holdout range")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(DEFAULT_DIST_DB))
    args = ap.parse_args()

    boards = board_lookup(args.max_seed, args.collected)
    con = sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True)
    rows = con.execute("SELECT clue, candidates, number, ranking FROM responses "
                       "WHERE model=? AND number IS NOT NULL", (args.teacher,)).fetchall()
    con.close()
    todo = []
    for clue, cand_j, number, rank_j in rows:
        cand, rank = json.loads(cand_j), json.loads(rank_j)
        if sorted(cand) != sorted(rank) or resolve_seed(rank, boards) is None:
            continue
        depth = min(int(number), len(cand) - 1)
        todo.append((clue, cand, int(number), rank[:depth - 1], depth))
    store = DistributionStore(args.out)
    todo = [t for t in todo if not store.has(args.lm, t[0], t[1], t[2], t[3])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} cached rankings from {args.teacher}; {len(todo)} positions to score", flush=True)

    scorer = LocalLMScorer(args.lm)
    t0 = time.time()
    for i, (clue, cand, number, path, depth) in enumerate(todo, 1):
        steps = scorer.path_distributions(clue, cand, number, path, depth)
        store.put(args.lm, clue, cand, number, path, steps)
        if i % 200 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - i) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
