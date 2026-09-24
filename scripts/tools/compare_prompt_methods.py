"""Do the four prompts in scripts/data/collect_prompt_variants.py pick the same
words? Free: reads cache/prompt_variants.db only.

**The distance.** For one position and step, let p_X be method X's pick
distribution. With n_X independent samples,

    within(X)  = (sum_w c_X(w)^2 - n_X) / (n_X (n_X - 1))    unbiased for sum p_X^2
    cross(X,Y) = sum_w p^_X(w) p^_Y(w)                       unbiased for sum p_X p_Y

so  D(X,Y) = within(X) + within(Y) - 2 cross(X,Y)  is an unbiased estimate of
||p_X - p_Y||^2, the squared L2 distance between the two distributions. It is 0
in expectation when the prompts do not differ, whatever the sampling noise --
which is the point of estimating it this way rather than comparing modes. It
can come out slightly negative on one position; the mean over positions is
what is reported, with a bootstrap interval over positions. For scale, two
distributions that put all their mass on different words have D = 2.

Also reported: `agree`, the chance that one draw from X equals one draw from Y
(cross), next to each method's self-agreement (within), and how often each
method's modal pick equals the first word of the ranking in llm_store.db,
which was bought without a temperature (near-greedy; see the collector).

Step 2 is conditioned on that stored first word. The single prompts removed it;
for the ranked prompts only samples whose own first word is it are kept, and
the kept fraction is reported, since a low one would make their step-2 row
thin.

**What it means for the listener.** `--booster` scores each method's picks
under a listener (McFadden R^2 against uniform over the words available at
that step), so "the prompts differ" can be read in the only unit that matters
here: would training labels from another prompt be predicted differently.

    python scripts/tools/compare_prompt_methods.py --booster cache/listener_gbt_oss_recipe.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = PROJECT_ROOT / "cache" / "prompt_variants.db"
HOLDOUT_START = 1_040_000
METHODS = {1: ["ranked", "ranked_nonumber", "single"],
           2: ["ranked", "ranked_nonumber", "single", "single_feedback"]}


def load(path: Path) -> dict:
    """{(seed, clue, number): {"first": w, "cand": [...], (method, step): [picks]}}"""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    pos: dict = defaultdict(lambda: defaultdict(list))
    for seed, clue, number, method, step, cand, removed, picks in conn.execute(
            "SELECT seed, clue, number, method, step, candidates, removed, picks FROM calls"):
        p = pos[(seed, clue, number)]
        picks, removed = json.loads(picks), json.loads(removed)
        if step == 1:
            p["cand"] = json.loads(cand)
            p[(method, 1, "list")].append(picks)
            p[(method, 1)].append(picks[0])
        else:
            p["first"] = removed[0]
            p[(method, 2)].append(picks[0])
    conn.close()
    # Ranked step 2: the second entry of those samples that began with the
    # stored first word -- the same conditioning the single prompts get.
    for p in pos.values():
        if "first" in p:
            for m in ("ranked", "ranked_nonumber"):
                lists = p.get((m, 1, "list"), [])
                p[(m, 2)] = [l[1] for l in lists if len(l) > 1 and l[0] == p["first"]]
                p[(m, 2, "kept")] = (len(p[(m, 2)]), len(lists))
    return pos


def within(picks: list[str]) -> float | None:
    n = len(picks)
    if n < 2:
        return None
    return (sum(c * c for c in Counter(picks).values()) - n) / (n * (n - 1))


def cross(a: list[str], b: list[str]) -> float | None:
    if not a or not b:
        return None
    ca, cb = Counter(a), Counter(b)
    return sum(ca[w] * cb[w] for w in ca) / (len(a) * len(b))


def boot(x: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tv(a: list[str], b: list[str]) -> float:
    """Total variation between two samples' empirical distributions: the share
    of probability that has to move to turn one into the other (0 same, 1
    disjoint)."""
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[w] / len(a) - cb[w] / len(b)) for w in set(ca) | set(cb))


def tv_with_null(a: list[str], b: list[str], rng, perms: int = 200) -> tuple[float, float, float]:
    """(observed TV, TV expected from sampling noise alone, permutation p).

    With 8 draws a side, two samples from the SAME distribution still differ
    -- plug-in TV is biased upward. The null is that bias measured: pool the
    draws, reshuffle them into groups of the same sizes, and recompute. What
    exceeds it is the part of the difference that belongs to the prompts."""
    obs = tv(a, b)
    pool = np.array(a + b, dtype=object)
    null = []
    for _ in range(perms):
        rng.shuffle(pool)
        null.append(tv(list(pool[:len(a)]), list(pool[len(a):])))
    null = np.array(null)
    return obs, float(null.mean()), float((1 + (null >= obs - 1e-12).sum()) / (1 + perms))


def direct_report(pos: dict, examples: int) -> None:
    """The distributions themselves, compared without any model."""
    rng = np.random.default_rng(0)
    print("\n\n######## Direct comparison of the pick distributions (no listener involved)")
    for step in (1, 2):
        ms = METHODS[step]
        print(f"\n== step {step}: shape of each prompt's distribution (mean over positions)")
        print(f"   {'prompt':17s} {'modal share':>12s} {'distinct words':>15s} {'entropy (bits)':>15s}")
        for m in ms:
            share, distinct, ent = [], [], []
            for p in pos.values():
                v = p.get((m, step), [])
                if len(v) < 2:
                    continue
                c = np.array(list(Counter(v).values()), dtype=float) / len(v)
                share.append(c.max())
                distinct.append(len(c))
                ent.append(float(-(c * np.log2(c)).sum()))
            print(f"   {m:17s} {np.mean(share):12.3f} {np.mean(distinct):15.2f} {np.mean(ent):15.3f}")
        print(f"\n== step {step}: total variation between prompts")
        print(f"   {'pair':34s} {'TV':>6s} {'noise':>6s} {'excess':>7s} {'95% CI':>17s} "
              f"{'same mode':>10s} {'p<0.05':>7s}  positions")
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                obs, null, sig, mode = [], [], [], []
                for p in pos.values():
                    va, vb = p.get((a, step), []), p.get((b, step), [])
                    if len(va) < 2 or len(vb) < 2:
                        continue
                    o, n, pv = tv_with_null(va, vb, rng)
                    obs.append(o); null.append(n); sig.append(pv < 0.05)
                    mode.append(Counter(va).most_common(1)[0][0] == Counter(vb).most_common(1)[0][0])
                ex = np.array(obs) - np.array(null)
                lo, hi = boot(ex)
                print(f"   {a + ' vs ' + b:34s} {np.mean(obs):6.3f} {np.mean(null):6.3f} {ex.mean():+7.3f} "
                      f"  [{lo:+.3f}, {hi:+.3f}] {np.mean(mode):10.3f} {np.mean(sig):7.3f}  {len(obs)}")
    print("\n   TV: share of probability that must move to turn one distribution into the other.")
    print("   noise: the TV two samples of the SAME distribution would show at these sample sizes.")
    print("   p<0.05: share of positions where the difference beats a permutation test; 0.05 if none differ.")

    if examples:
        print(f"\n== the {examples} step-2 positions where the list continuation and the re-ask differ most")
        scored = []
        for (seed, clue, number), p in pos.items():
            va, vb = p.get(("ranked", 2), []), p.get(("single", 2), [])
            if len(va) >= 4 and len(vb) >= 4:
                scored.append((tv(va, vb), clue, number, p))
        for d, clue, number, p in sorted(scored, key=lambda t: -t[0])[:examples]:
            fmt = lambda v: ", ".join(f"{w} {c}" for w, c in Counter(v).most_common())
            print(f"\n   clue '{clue}' for {number}, first word '{p['first']}' removed   (TV {d:.2f})")
            print(f"      step 1  ranked:          {fmt(p[('ranked', 1)])}")
            print(f"      step 2  ranked (cont.):  {fmt(p[('ranked', 2)])}")
            print(f"      step 2  single (re-ask): {fmt(p[('single', 2)])}")
            print(f"      step 2  single_feedback: {fmt(p[('single_feedback', 2)])}")


def listener_r2(pos: dict, booster_path: Path) -> None:
    import lightgbm as lgb

    from codenames.listener_training import DB, DEFAULT_MODEL, load_positions

    booster = lgb.Booster(model_file=str(booster_path))
    lp, _ = load_positions(DB, DEFAULT_MODEL, 0, 45000, seed_filter=lambda s: s >= HOLDOUT_START)
    by_key = {(p["key"][0], p["key"][2], frozenset(w.lower() for w in p["key"][1])): p for p in lp}
    ll: dict = defaultdict(list)
    null: dict = defaultdict(list)
    for (seed, clue, number), p in pos.items():
        rec = by_key.get((clue, number, frozenset(w.lower() for w in p["cand"])))
        if rec is None:
            continue
        s = booster.predict(rec["x"], raw_score=True)
        row = {w.lower(): r for r, w in enumerate(rec["words"])}
        for step in (1, 2):
            if step == 2 and "first" not in p:
                continue
            keep = [r for r in range(rec["n"]) if step == 1 or r != row[p["first"].lower()]]
            z = s[keep] - s[keep].max()
            logp = z - np.log(np.exp(z).sum())
            where = {r: i for i, r in enumerate(keep)}
            for m in METHODS[step]:
                for w in p.get((m, step), []):
                    ll[(m, step)].append(-logp[where[row[w.lower()]]])
                    null[(m, step)].append(np.log(len(keep)))
    print(f"\nlistener {booster_path.name}: McFadden R^2 of each prompt's picks")
    for step in (1, 2):
        for m in METHODS[step]:
            a, b = np.array(ll[(m, step)]), np.array(null[(m, step)])
            if len(a):
                print(f"   step {step}  {m:17s} {1 - a.mean() / b.mean():7.4f}   ({len(a)} picks)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=VARIANTS)
    ap.add_argument("--booster", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=5, help="step-2 positions to print side by side")
    args = ap.parse_args()

    pos = load(args.db)
    print(f"{len(pos)} positions, {sum('first' in p for p in pos.values())} with step 2")
    for step in (1, 2):
        ms = METHODS[step]
        print(f"\n== step {step}")
        if step == 2:
            for m in ("ranked", "ranked_nonumber"):
                k = np.array([p[(m, 2, "kept")] for p in pos.values() if (m, 2, "kept") in p])
                print(f"   {m}: {k[:, 0].sum()}/{k[:, 1].sum()} samples began with the stored first word "
                      f"({k[:, 0].sum() / k[:, 1].sum():.0%})")
        print(f"   {'self-agreement':34s}" + "".join(
            f"{m}: {np.mean([v for p in pos.values() if (v := within(p.get((m, step), []))) is not None]):.3f}   "
            for m in ms))
        print(f"   {'pair':34s} {'agree':>7s} {'D = ||pX-pY||^2':>16s} {'95% CI':>18s}  positions")
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                d, ag = [], []
                for p in pos.values():
                    wa, wb, c = within(p.get((a, step), [])), within(p.get((b, step), [])), \
                        cross(p.get((a, step), []), p.get((b, step), []))
                    if None in (wa, wb, c):
                        continue
                    d.append(wa + wb - 2 * c)
                    ag.append(c)
                d = np.array(d)
                lo, hi = boot(d)
                print(f"   {a + ' vs ' + b:34s} {np.mean(ag):7.3f} {d.mean():16.4f} "
                      f"   [{lo:.4f}, {hi:.4f}]  {len(d)}")
        if step == 1:
            print("   modal pick == llm_store.db's first word (bought without a temperature):")
            for m in ms:
                hits = [Counter(p[(m, 1)]).most_common(1)[0][0] == p["first"]
                        for p in pos.values() if p.get((m, 1)) and "first" in p]
                if hits:
                    print(f"      {m:17s} {np.mean(hits):.3f}  ({len(hits)})")
    direct_report(pos, args.examples)
    if args.booster:
        listener_r2(pos, args.booster)


if __name__ == "__main__":
    main()
