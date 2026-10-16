"""Can the stop listener stand in for gpt-oss's "sure" answer? Free (no API).

The clarity knob (docs/log.md, "Comfort probe") needs to score thousands of
candidate clues per board for how sure a guesser would be, which cannot be a
gpt-oss call each. The stop listener already predicts, without an API call,
which words gpt-oss's relatedness prompt would put in its "guess" group
(strict model) or in "guess" + "stretch" (loose model). This checks how well
that predicts the separate "sure" question, on the turns of a comfort probe
run (scripts/tools/probe_comfort.py, saved in cache/report/).

For each turn and each stop model, at pick 1:
- `count`: words scored above STOP;
- `soft`: the sum over words of sigmoid(score - STOP score), a smooth count.
Each is capped at the number and divided by it, like sure / number.

Reported: Spearman correlation with sure / number over turns (what the knob
needs is the ranking of clues), the overlap of the listener's set with the
sure words, AUC for "gpt-oss found all N" beside the sure answer's own, and
the per-spymaster means (does the proxy order the models as gpt-oss does).

    python scripts/tools/check_sure_proxy.py cache/report/comfort_probe_300_sure.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    for r, x in ((ra, np.asarray(a)), (rb, np.asarray(b))):
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, y) -> float:
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()

    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.stop_listener import StopListenerSpymaster

    sims = SimilarityTensor.load()
    stops = {"strict": StopListenerSpymaster(stop_model_path=DEFAULT_CACHE_DIR / "listener_gbt_stop_guess.txt"),
             "loose": StopListenerSpymaster()}
    turns = [t for t in json.loads(args.report.read_text())["turns"] if (t.get("answers") or {}).get("sure")]

    rows = []
    for t in turns:
        k, words = t["k"], t["words"]
        sure = set(t["answers"]["sure"]["sure"])
        row = {"spymaster": t["spymaster"], "k": k, "sure": min(len(sure), k) / k, "all_found": t["all_found"]}
        ok = True
        for name, sm in stops.items():
            v = sm.stop_scores([t["clue"]], words, k, sims).get(0)
            if v is None:
                ok = False
                break
            s, s_stop = v[0, :-1], v[0, -1]
            above = {words[i] for i in np.flatnonzero(s > s_stop)}
            row[f"{name} count"] = min(len(above), k) / k
            row[f"{name} soft"] = min(float(np.sum(1 / (1 + np.exp(-(s - s_stop))))), k) / k
            row[f"{name} overlap"] = (len(above & sure), len(above), len(sure))
        if ok:
            rows.append(row)

    print(f"{len(rows)} of {len(turns)} turns scored\n")
    sure = np.array([r["sure"] for r in rows])
    y = np.array([r["all_found"] for r in rows])
    print(f"{'proxy':16s} {'Spearman vs sure/k':>19s} {'AUC all found':>14s}   set overlap with sure words")
    print(f"{'sure/k (gpt-oss)':16s} {'1.000':>19s} {auc(sure, y):14.3f}")
    out = {"n": len(rows), "auc_sure": auc(sure, y), "proxies": {}}
    for name in stops:
        for kind in ("count", "soft"):
            x = np.array([r[f"{name} {kind}"] for r in rows])
            rho, a = spearman(x, sure), auc(x, y)
            line = f"{name + ' ' + kind:16s} {rho:19.3f} {a:14.3f}"
            if kind == "count":
                both, n_above, n_sure = (np.sum([r[f"{name} overlap"][i] for r in rows]) for i in range(3))
                line += f"   {both / n_above:.0%} of its words are sure; it has {both / n_sure:.0%} of the sure words"
            print(line)
            out["proxies"][f"{name} {kind}"] = {"spearman": rho, "auc": a}

    print("\nBy spymaster: mean sure/k (gpt-oss) and each proxy")
    names = list(dict.fromkeys(r["spymaster"] for r in rows))
    cols = ["sure"] + [f"{n} {k}" for n in stops for k in ("count", "soft")]
    print(f"   {'':40s} " + " ".join(f"{c:>12s}" for c in cols))
    out["by_spymaster"] = {}
    for n in names:
        rs = [r for r in rows if r["spymaster"] == n]
        vals = [float(np.mean([r[c] for r in rs])) for c in cols]
        out["by_spymaster"][n] = dict(zip(cols, vals))
        print(f"   {n:40s} " + " ".join(f"{v:12.3f}" for v in vals))
    path = PROJECT_ROOT / "cache" / "report" / f"sure_proxy_{args.report.stem}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
