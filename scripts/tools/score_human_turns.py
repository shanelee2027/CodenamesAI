"""How well do candidate guesser models predict a human guesser? Scored on
the /eval page's recorded turns (cache/human_eval.jsonl), at no API cost.

Step 1 of the comfort work (docs/log.md, "A coarse guesser"): a guesser
model that plays like a human is what lets a comfort weighting be chosen by
simulation instead of by hand. This asks which of our models is closest to
the one human we have data for.

A turn is: board (seed), pre-revealed words, clue, number k, the words picked
in order, and how it ended (a wrong word, or the Stop button). It is scored
in two separate parts, so that models with and without a stop can be
compared on the part they share.

1. **Which word, given the guesser goes on.** At each pick j, the log
   probability of the word actually picked among the words left. Reported as
   McFadden R² against uniform over the words left, by pick.
2. **Whether it stops before k.** A decision point is every pick j <= k
   reached with all earlier picks own: the human either picked (went on) or
   pressed Stop. Stopping at k is not a decision (the page allows a k+1st
   guess, which was never used), nor is any point after a wrong word.

Models:
- `frozen`: assoc_profile booster scores, softmax over the words left (what
  the incumbent assumes).
- `temperatures`: the same at temperature tau_j (profile_temperature_listener).
- `stop` / `stop strict`: the stop listener's rows (words + STOP, trained on
  gpt-oss's "which words does this clue point at" labels, guess+stretch or
  guess only). Part 1 uses its word softmax with STOP removed; part 2 its
  P(STOP).
- `coarse m, eps`: ranks only its first m picks (as `frozen`), then picks
  uniformly among the related words left, where "related" is the stop
  listener's set at pick 1 (words scored above STOP); eps of the
  probability goes uniformly to the rest. It stops when no related word is
  left (with probability 1 - eps). This is the "rank the top one or two,
  then related-or-not" guesser.

eps and m are not fitted on the human turns; a few values are shown.

    python scripts/tools/score_human_turns.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = PROJECT_ROOT / "cache" / "human_eval.jsonl"
TEMPERATURES = PROJECT_ROOT / "cache" / "sequential_listener_assoc_profile_temperature.json"
COARSE = [(1, 0.05), (1, 0.2), (2, 0.05), (2, 0.2)]


def softmax_logp(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()

    from codenames.board import Board, Role
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.assoc_profile_listener import AssocProfileListenerSpymaster
    from codenames.spymasters.stop_listener import StopListenerSpymaster

    sims = SimilarityTensor.load()
    base = AssocProfileListenerSpymaster()
    stops = {"stop": StopListenerSpymaster(),
             "stop strict": StopListenerSpymaster(stop_model_path=DEFAULT_CACHE_DIR / "listener_gbt_stop_guess.txt")}
    a = json.loads(TEMPERATURES.read_text())["a"]
    tau = [1.0] + [float(np.exp(-x)) for x in a]                 # picks 1, 2, 3, 4+

    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    word_ll: dict[str, list] = {}          # model -> [(pick j, logp, log n_left)]
    stop_ev: dict[str, list] = {}          # model -> [(p_stop, stopped)]
    skipped = 0
    for r in rows:
        board = Board.generate(seed=r["seed"])
        pre = {w.lower() for w in r["pre_revealed"]}
        left = [c.word for c in board.cards if c.word.lower() not in pre]
        own = {c.word for c in board.cards if c.role is Role.OWN}
        picks = [p["word"] for p in r["picks"]]
        k, clue = int(r["number"]), r["clue"]
        f = base._listener_features(clue, left, k, sims)
        if f is None or any(p not in left for p in picks):
            skipped += 1
            continue
        s = np.asarray(base.bundle.booster.predict(f, raw_score=True), dtype=np.float64)
        S = {name: sm.stop_scores([clue], left, k, sims).get(0) for name, sm in stops.items()}
        if any(v is None for v in S.values()):
            skipped += 1
            continue
        related = {name: {left[i] for i in np.flatnonzero(v[0, :-1] > v[0, -1])} for name, v in S.items()}

        # Decision points: before pick j = 1..min(len(picks) + 1, k), all earlier picks own.
        stopped_early = r["ended_by"] == "stopped" and len(picks) < k
        n_dec = len(picks) + 1 if stopped_early else min(len(picks), k)
        gone: list[str] = []
        for j in range(1, n_dec + 1):
            idx = [i for i, w in enumerate(left) if w not in gone]
            names = [left[i] for i in idx]
            is_stop = j == len(picks) + 1                    # the human pressed Stop here
            if not is_stop:
                w = picks[j - 1]
                t = names.index(w)
                null = np.log(len(names))
                lp = softmax_logp(s[idx])
                word_ll.setdefault("frozen", []).append((j, lp[t], null))
                lpt = softmax_logp(s[idx] / tau[min(j, 4) - 1])
                word_ll.setdefault("temperatures", []).append((j, lpt[t], null))
                for name, v in S.items():
                    row = v[j - 1]
                    word_ll.setdefault(name, []).append((j, softmax_logp(row[:-1][idx])[t], null))
                for m, eps in COARSE:
                    rel = [i for i, x in enumerate(names) if x in related["stop"]]
                    if j <= m:
                        p = np.exp(lp[t])
                    else:
                        inside = w in related["stop"]
                        p = ((1 - eps) / len(rel) if inside else eps / max(len(names) - len(rel), 1)) if rel \
                            else 1 / len(names)
                    word_ll.setdefault(f"coarse m={m}, eps={eps}", []).append((j, np.log(p), null))
            # Stop decision (only before k).
            if j <= k and (j <= len(picks) or is_stop):
                for name, v in S.items():
                    stop_ev.setdefault(name, []).append((float(np.exp(softmax_logp(np.append(v[j - 1][:-1][idx], v[j - 1][-1]))[-1])), is_stop))
                for eps in (0.05, 0.2):
                    none_left = not any(x in related["stop"] for x in names)
                    stop_ev.setdefault(f"coarse eps={eps}", []).append((1 - eps if none_left else eps, is_stop))
            if not is_stop:
                gone.append(picks[j - 1])

    print(f"{len(rows) - skipped} turns scored ({skipped} skipped)\n")
    print("1. Which word, given the guesser goes on: McFadden R² against uniform over the words left")
    print(f"   {'model':24s} {'all':>7s} {'pick 1':>7s} {'pick 2':>7s} {'pick 3+':>7s}   events")
    out = {"word": {}, "stop": {}}
    for name, ev in word_ll.items():
        e = np.array(ev, dtype=float)
        r2 = lambda m: 1 - (-e[m, 1].sum()) / e[m, 2].sum()
        cols = [np.ones(len(e), bool), e[:, 0] == 1, e[:, 0] == 2, e[:, 0] >= 3]
        vals = [r2(c) for c in cols]
        out["word"][name] = vals
        print(f"   {name:24s} " + " ".join(f"{v:7.3f}" for v in vals) + f"   {len(e)}")

    print("\n2. Stop before k: log-loss per decision (lower is better), AUC, mean P(stop) when it stopped / went on")
    rate = np.mean([st for _, st in next(iter(stop_ev.values()))])
    ys = np.array([st for _, st in next(iter(stop_ev.values()))], dtype=float)
    base_ll = -np.mean(ys * np.log(rate) + (1 - ys) * np.log(1 - rate))
    print(f"   {'base rate (fitted)':24s} {base_ll:7.3f}   AUC 0.500   {rate:.3f} / {rate:.3f}   "
          f"({int(ys.sum())} stops in {len(ys)} decisions)")
    for name, ev in stop_ev.items():
        p = np.clip(np.array([x for x, _ in ev]), 1e-6, 1 - 1e-6)
        y = np.array([st for _, st in ev], dtype=float)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        pos, neg = p[y == 1], p[y == 0]
        auc = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
        out["stop"][name] = {"logloss": ll, "auc": auc}
        print(f"   {name:24s} {ll:7.3f}   AUC {auc:.3f}   {pos.mean():.3f} / {neg.mean():.3f}")
    path = PROJECT_ROOT / "cache" / "report" / "human_turns_guesser_models.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
