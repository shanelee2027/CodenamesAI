"""How often is the GBT spymaster's best clue over the WHOLE clue pool inside the Gaussian shortlist?

States: every turn the GBT spymaster (fresh_split_listener) faced in the match `gbt-tunetemps-vs-gauss15-test-p1`
(300 test-word boards, both seatings), rebuilt from the game records: the board from its seed and the test words,
every word guessed before that turn revealed, seen from the GBT's side.

Per state:
- the Gaussian first stage's scores over the pool (what the shortlist is cut from, illegal clues included, exactly
  as learned_listener does it);
- the GBT's expected reward for EVERY clue the Gaussian scores (shortlist = whole pool);
- the full-pool pick (legal, same tie-break as play) and its rank in the Gaussian ordering;
- for each shortlist size N: the pick the GBT makes from the top N, and its expected reward under the GBT.
One JSON line per state, appended, so a restart resumes.

    python shortlist_coverage.py --workers 8 --out cache/shortlist_coverage.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/maste/CodenamesAI")
LABEL = "gbt-tunetemps-vs-gauss15-test-p1"
SIZES = (25, 50, 100, 200, 300, 500, 1000, 2000, 5000)
G = {}


def states():
    from codenames.board import Board, load_three_way_split

    vocab = load_three_way_split()["test"]
    c = sqlite3.connect(f"file:{ROOT / 'cache/llm_store.db'}?mode=ro", uri=True)
    out = []
    for rid, seed, label, turns in c.execute(
            "select id, seed, label, turns from game_records where label like ? order by id", (LABEL + "|%",)):
        gbt = "A" if label.split("|A=")[1].startswith("fresh_split_listener") else "B"
        words = set(Board.generate(seed=seed, vocabulary=vocab).words)
        revealed = []
        for ti, t in enumerate(json.loads(turns)):
            if t["team"] == gbt:
                out.append({"key": f"{rid}:{ti}", "seed": seed, "side": gbt, "turn": ti, "revealed": list(revealed),
                            "clue_played": t["clue"], "number_played": t["number"]})
            for w, _ in t["guesses"]:
                assert w in words, (seed, w)
                revealed.append(w)
    return out


def init():
    from codenames.board import load_three_way_split
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    G["sims"] = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    G["vocab"] = load_three_way_split()["test"]
    cls, params = spymaster_spec("fresh_split_listener", shortlist=10**9)
    G["sm"] = cls(**params)


def run(st):
    from codenames.board import Board, OpponentBoardView

    t0 = time.perf_counter()
    sims, sm = G["sims"], G["sm"]
    board = Board.generate(seed=st["seed"], vocabulary=G["vocab"])
    board.revealed = set(st["revealed"])
    view = board if st["side"] == "A" else OpponentBoardView(board)
    fs = sm._first_stage
    g_n, g, g_margin = fs._score_all_clues(view, sims)
    best_n, scores, margin = sm._score_all_clues(view, sims)          # GBT over the whole pool
    finite = np.flatnonzero(np.isfinite(g))
    g_order = finite[np.argsort(-g[finite], kind="stable")]           # the order the shortlist is cut from
    g_rank = np.full(len(g), -1)
    g_rank[g_order] = np.arange(1, len(g_order) + 1)
    full = fs._pick_top_clues(sims, view, best_n, scores, margin, 1)[0]
    ci = sims.clue_index[full[0].lower()]
    rec = {**st, "n_scored": int(np.isfinite(scores).sum()), "full": {"clue": full[0], "number": full[1],
           "value": full[2], "g_rank": int(g_rank[ci])}, "by_size": {}}
    for n in SIZES:
        keep = np.full(len(scores), -np.inf, dtype=scores.dtype)
        idx = g_order[:n]
        keep[idx] = scores[idx]
        pick = fs._pick_top_clues(sims, view, best_n, keep, margin, 1)[0]
        rec["by_size"][n] = {"clue": pick[0], "number": pick[1], "value": pick[2]}
    rec["seconds"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "cache/shortlist_coverage.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in states() if s["key"] not in done][: args.limit]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=init) as pool, out.open("a") as f:
        for i, rec in enumerate(pool.imap_unordered(run, todo, chunksize=1), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)}  {el / 60:.1f} min  eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
