"""How overpromise_listener's clues change with its promise cost λ
(docs/versions/overpromise_listener.md), on many positions at once.

The cost enters a turn's value linearly: value_k(λ) = A_k − λ·B_k, where A_k
is the expected net words of a clue for k and B_k the expected number of
promised words the guesser stops short of. So each position needs ONE search:
the shortlist, the STOP listener's scores, and A and B for every clue and
number. The clue played at any λ is then the model's own final selection
(`_pick_top_clues`) over A − λB, exactly what the spymaster does, checked
against it on the first positions.

Positions are the compare page's (play_server.deal_position: a fresh board
with 0-8 cards revealed, at least 2 own words left), from a fixed seed.

    python scripts/tools/sweep_promise_cost.py                       # guess + stretch, 400 positions
    python scripts/tools/sweep_promise_cost.py --stop-model listener_gbt_stop_guess.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
CHECK = 5                       # positions on which the selection is checked against the spymaster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stop-model", default="listener_gbt_stop.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2, type=float, default=(0.5, 1.0), help="two λ to list differences for")
    args = ap.parse_args()

    from play_server import deal_position

    from codenames.board import clue_number_cap
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.base import TurnContext
    from codenames.spymasters.registry import spymaster_spec
    from codenames.spymasters.stop_listener import StopListenerSpymaster
    from codenames.spymasters.stop_net_words_listener import turn_net_words

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    cls, kw = spymaster_spec("overpromise_listener", stop_model_path=DEFAULT_CACHE_DIR / args.stop_model)
    sm = cls(**kw)
    lams = sorted(set(LAMBDAS) | set(args.compare))
    index = {w: i for i, w in enumerate(sims.clue_words)}
    random.seed(args.seed)
    rows, t0 = [], time.time()
    for n in range(args.n):
        seed, board, pre = deal_position()
        best_n, scores, margin = super(StopListenerSpymaster, sm)._score_all_clues(board, sims)
        own, words, roles = sm._board(board)
        finite = np.flatnonzero(np.isfinite(scores))
        cap = clue_number_cap(len(own), sm.max_number)
        costs = np.array([sm.costs[r] for r in roles], dtype=np.float64)
        S = sm.stop_scores([sims.clue_words[i] for i in finite], words, cap, sims)
        A, B, first = {}, {}, {}
        for t, s in S.items():
            a = turn_net_words(s, len(own), costs, cap, 0.0)
            A[finite[t]], B[finite[t]] = a, a - turn_net_words(s, len(own), costs, cap, 1.0)
            first[finite[t]] = s[0, :len(own)]
        row = {"seed": seed, "revealed": len(pre), "own_left": len(own), "choices": {}}
        for lam in lams:
            sc, bn = np.full_like(scores, -np.inf), best_n.copy()
            for ci in A:
                v = A[ci] - lam * B[ci]
                sc[ci], bn[ci] = v.max(), int(v.argmax()) + 1
            clue, k, _ = sm._first_stage._pick_top_clues(sims, board, bn, sc, margin, 1)[0]
            ci = index[clue]
            order = np.argsort(-first[ci])
            row["choices"][str(lam)] = {"clue": clue, "number": int(k), "net_words": float(A[ci][k - 1]),
                                        "shortfall": float(B[ci][k - 1]),
                                        "targets": [own[i] for i in order[:k]]}
        if n < CHECK:                           # the selection here is the spymaster's own
            sm.promise_cost = args.compare[0]
            got = sm.top_clues(TurnContext(board=board, turn_index=len(pre)), sims, 1)[0][:2]
            c = row["choices"][str(args.compare[0])]
            assert (c["clue"], c["number"]) == tuple(got), (got, c)
        rows.append(row)
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {n + 1}/{args.n} positions, {el / (n + 1):.1f}s each, ~{el / (n + 1) * (args.n - n - 1) / 60:.0f} min left",
                  flush=True)

    print(f"\n{args.n} positions (deal_position, seed {args.seed}), STOP listener {args.stop_model}")
    print(f"  {'λ':>5s}  {'mean k':>6s}  {'k=1':>5s} {'k=2':>5s} {'k=3':>5s} {'k=4':>5s}   {'net words':>9s}  "
          f"{'shortfall':>9s}  {'differs from λ=' + str(args.compare[0]):>20s}")
    ref = args.compare[0]
    summary = {}
    for lam in lams:
        ks = np.array([r["choices"][str(lam)]["number"] for r in rows])
        nw = np.mean([r["choices"][str(lam)]["net_words"] for r in rows])
        sf = np.mean([r["choices"][str(lam)]["shortfall"] for r in rows])
        diff = np.mean([(r["choices"][str(lam)]["clue"], r["choices"][str(lam)]["number"])
                        != (r["choices"][str(ref)]["clue"], r["choices"][str(ref)]["number"]) for r in rows])
        summary[str(lam)] = {"mean_number": float(ks.mean()), "dist": {k: float((ks == k).mean()) for k in (1, 2, 3, 4)},
                             "net_words": float(nw), "shortfall": float(sf), "differs_from_ref": float(diff)}
        print(f"  {lam:5.2f}  {ks.mean():6.2f}  " + " ".join(f"{(ks == k).mean():5.0%}" for k in (1, 2, 3, 4))
              + f"   {nw:9.3f}  {sf:9.3f}  {diff:20.0%}")

    a, b = (str(x) for x in args.compare)
    d = [r for r in rows if (r["choices"][a]["clue"], r["choices"][a]["number"])
         != (r["choices"][b]["clue"], r["choices"][b]["number"])]
    same_clue = sum(r["choices"][a]["clue"] == r["choices"][b]["clue"] for r in d)
    print(f"\nλ={a} vs λ={b}: {len(d)} of {len(rows)} positions differ ({len(d) / len(rows):.0%}); "
          f"{same_clue} of those keep the clue and change only the number")
    for r in d[:25]:
        ca, cb = r["choices"][a], r["choices"][b]
        print(f"  {r['own_left']} own left: {ca['clue']} {ca['number']} ({', '.join(ca['targets'])})  ->  "
              f"{cb['clue']} {cb['number']} ({', '.join(cb['targets'])})   "
              f"net {ca['net_words']:.2f} -> {cb['net_words']:.2f}, shortfall {ca['shortfall']:.2f} -> {cb['shortfall']:.2f}")
    out = PROJECT_ROOT / "cache" / "report" / f"promise_cost_sweep_{Path(args.stop_model).stem}.json"
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
