"""How much of gpt-oss's choice could any listener predict? (docs/log.md,
"How much is left to explain: the ceiling").

Free: reuses cache/prompt_variants.db (scripts/data/collect_prompt_variants.py),
8 rankings per position at temperature 1 under the training prompt, the same
clue, candidates and order every time, for 300 clean-holdout positions. With
repeated draws from one fixed prompt, the spread between them is the
guesser's own randomness, which no feature can explain.

The ceiling is bracketed from both sides, per pick:
- **plug-in (upper bound on R², accuracy):** the empirical distribution of the
  8 draws scored on those same draws. Overfits by construction, so no model
  can beat it on average.
- **leave-one-out (achievable):** each draw predicted from the other 7
  (add-0.5 smoothing over the candidates; accuracy from their modal word).
  A real, if crude, predictor that sees many samples of the truth, so the
  ceiling is at least this.

Pick 2 is the list's second entry, conditioned on the first being the
position's modal first word (the draws that took it; positions with at least
4 such draws).

The listeners are scored on the same draws with frozen scores, as the reward
uses them. Caveat: the training labels are near-greedy (no temperature sent),
and these are temperature-1 draws, so this is the ceiling for predicting a
sampling guesser. A near-greedy guesser is more predictable, so its ceiling is
if anything higher.

    python scripts/tools/listener_ceiling.py [--models cache/listener_gbt.txt cache/listener_gbt_isa.txt]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

DB = Path("cache/prompt_variants.db")


def draws() -> dict:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out: dict = {}
    for seed, clue, number, cands, picks in c.execute(
            "select seed, clue, number, candidates, picks from calls where method='ranked' and step=1"):
        cands, picks = json.loads(cands), json.loads(picks)
        if len(picks) < 2:
            continue
        out.setdefault((seed, clue, number, tuple(cands)), []).append(picks)
    return out


def bounds(samples: list[str], n: int) -> tuple[float, float, float, float]:
    """(plug-in NLL, LOO NLL, plug-in accuracy, LOO accuracy) for one
    position's draws over n candidates."""
    m = len(samples)
    cnt = Counter(samples)
    plug_nll = -np.mean([np.log(cnt[s] / m) for s in samples])
    plug_acc = max(cnt.values()) / m
    loo_nll, loo_acc = [], []
    for s in samples:
        c = cnt.copy()
        c[s] -= 1
        loo_nll.append(-np.log((c[s] + 0.5) / (m - 1 + 0.5 * n)))
        top = max(c.values())
        modal = [w for w, v in c.items() if v == top]
        loo_acc.append((s in modal) / len(modal))
    return plug_nll, float(np.mean(loo_nll)), plug_acc, float(np.mean(loo_acc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", type=Path,
                    default=[Path("cache/listener_gbt.txt"), Path("cache/listener_gbt_isa.txt")])
    args = ap.parse_args()

    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.learned_listener import LearnedListenerSpymaster

    D = draws()
    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    listeners = {p.name: LearnedListenerSpymaster(model_path=p) for p in args.models}

    rows = {1: [], 2: []}          # (plug nll, loo nll, plug acc, loo acc, log n, {model: (nll, acc)})
    for (seed, clue, number, cands), rk in D.items():
        cands = list(cands)
        n = len(cands)
        scores = {}
        for name, sm in listeners.items():
            f = sm._listener_features(clue, cands, number, sims)
            if f is None:
                break
            scores[name] = sm.bundle.booster.predict(f, raw_score=True)
        if len(scores) < len(listeners):
            continue
        idx = {w: i for i, w in enumerate(cands)}

        def model_stats(targets: list[str], removed: str | None):
            out = {}
            for name, s in scores.items():
                z = s.copy()
                if removed is not None:
                    z[idx[removed]] = -np.inf
                lp = z - np.logaddexp.reduce(z[np.isfinite(z)])
                top = np.argmax(lp)
                out[name] = (-np.mean([lp[idx[t]] for t in targets]),
                             np.mean([float(idx[t] == top) for t in targets]))
            return out

        first = [r[0] for r in rk]
        rows[1].append((*bounds(first, n), np.log(n), model_stats(first, None)))
        modal = Counter(first).most_common(1)[0][0]
        second = [r[1] for r in rk if r[0] == modal]
        if len(second) >= 4:
            rows[2].append((*bounds(second, n - 1), np.log(n - 1), model_stats(second, modal)))

    for step, rs in rows.items():
        null = np.mean([r[4] for r in rs])
        r2 = lambda k: 1 - np.mean([r[k] for r in rs]) / null
        print(f"\npick {step}: {len(rs)} positions, {sum(1 for _ in rs)} x ~{8 if step == 1 else 6} draws")
        print(f"  {'':34s} {'R2':>7s} {'accuracy':>9s}")
        print(f"  {'ceiling, plug-in (upper bound)':34s} {r2(0):7.3f} {np.mean([r[2] for r in rs]):9.3f}")
        print(f"  {'ceiling, leave-one-out (reached)':34s} {r2(1):7.3f} {np.mean([r[3] for r in rs]):9.3f}")
        for name in listeners:
            nll = np.mean([r[5][name][0] for r in rs])
            acc = np.mean([r[5][name][1] for r in rs])
            print(f"  {'listener ' + name:34s} {1 - nll / null:7.3f} {acc:9.3f}")


if __name__ == "__main__":
    main()
