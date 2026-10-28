"""Which board words the listener booster gets most wrong, and whether they
share a category (a location, an animal, ...) that a feature could flag.
Costs nothing -- no LLM.

The booster is scored on held-out positions, one Plackett-Luce event per
pick, as in eval_listener_accuracy.py. For each event, every candidate word
w gets its predicted probability p_w and the label y_w (1 if picked).
Per word and per category, summed over every event it was a candidate in:
- *picks* O = sum y and *expected* E = sum p. O/E > 1: the guesser picks it
  more often than the booster says; < 1, less. A flag only helps if O/E
  departs from 1 for a whole category;
- *surprise*: the mean -log p over the events it was picked in, against the
  mean over all events. High: when it is picked, the booster had it low.

Categories are WordNet lexicographer files (noun.location, noun.animal, ...)
of the word's first noun sense, so "the word is a location" is its most
common reading. Sense counts (all parts of speech) test polysemy. The clue is categorised the same way, for the case the user
raised: a category clue (ASIA, ANIMAL) whose meaning is unclear.

    python scripts/tools/word_loss.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path(__file__).resolve().parents[2] / "cache"
BOOSTER = "listener_gbt_assoc_profile.txt"     # the incumbent's and floored_overpromise_listener's


def category(word: str) -> str:
    from nltk.corpus import wordnet as wn

    s = wn.synsets(word.lower().replace(" ", "_"), pos=wn.NOUN)
    return s[0].lexname().removeprefix("noun.") if s else "(no noun)"


def n_senses(word: str) -> str:
    from nltk.corpus import wordnet as wn

    n = len(wn.synsets(word.lower().replace(" ", "_")))
    return "1-2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


def events(booster, positions: list[dict]):
    """Per event: board seed, pick index, clue, candidate words, p, y."""
    from train_listener_net import gbt_scores

    out = []
    for p, z in zip(positions, gbt_scores(booster, positions)):
        taken: list[int] = []
        for j in range(min(p["k"], p["n"] - 1)):
            keep = [r for r in range(p["n"]) if r not in taken]
            zk = z[keep] - z[keep].max()
            pk = np.exp(zk) / np.exp(zk).sum()
            t = p["targets"][j]
            out.append((p["seed"], j, p["clue"], [p["words"][r] for r in keep], pk, keep.index(t)))
            taken.append(t)
    return out


def table(ev, key, min_events: int, rng, top: int | None = None, sort: str = "oe"):
    """O/E and surprise grouped by key(clue, word), with a 95% bootstrap over
    boards on O/E."""
    rows = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))   # group -> board -> [O, E, nll sum, n picked]
    nll_all = []
    for seed, j, clue, words, p, t in ev:
        nll_all.append(-np.log(p[t]))
        for i, w in enumerate(words):
            g = key(clue, w)
            if g is None:
                continue
            a = rows[g][seed]
            a[0] += i == t
            a[1] += p[i]
            if i == t:
                a[2] += -np.log(p[t])
                a[3] += 1
    base = float(np.mean(nll_all))
    res = []
    for g, boards in rows.items():
        m = np.array(list(boards.values()))
        if m[:, 1].sum() < min_events * 0.04 or m[:, 3].sum() < min_events:
            continue
        idx = rng.integers(0, len(m), (1000, len(m)))
        oe = m[idx, 0].sum(1) / m[idx, 1].sum(1)
        lo, hi = np.percentile(oe, [2.5, 97.5])
        res.append((g, int(m[:, 3].sum()), m[:, 0].sum() / m[:, 1].sum(), lo, hi, m[:, 2].sum() / m[:, 3].sum()))
    res.sort(key=(lambda r: abs(np.log(r[2]))) if sort == "oe" else (lambda r: -r[5]), reverse=sort == "oe")
    return base, res[:top] if top else res


def show(title, base, res):
    print(f"\n{title}  (mean surprise over all events {base:.2f})")
    print(f"  {'group':34s} {'picks':>6s} {'O/E':>6s} {'95% CI':>15s} {'surprise':>9s}")
    for g, n, oe, lo, hi, s in res:
        flag = "" if lo <= 1 <= hi else " *"
        print(f"  {str(g):34s} {n:6d} {oe:6.2f} [{lo:5.2f}, {hi:5.2f}] {s:9.2f}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", default=BOOSTER)
    ap.add_argument("--sets", nargs="+", default=["val", "held-out words, generated",
                                                  "held-out words, generated, Sonnet 5.5"])
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    import lightgbm as lgb

    from eval_listener_accuracy import with_all_columns
    from train_listener_net import load_sets

    booster = lgb.Booster(model_file=str(CACHE / args.booster))
    sets = load_sets()
    rng = np.random.default_rng(0)
    cat = {}
    cat_of = lambda w: cat.setdefault(w, category(w))
    for s in args.sets:
        ev = events(booster, with_all_columns(sets[s]))
        print(f"\n===== {s}: {len(ev)} events, {len({e[0] for e in ev})} boards")
        show("Board words, furthest from O/E = 1", *table(ev, lambda c, w: w, 15, rng, args.top))
        show("Board words, highest surprise when picked", *table(ev, lambda c, w: w, 15, rng, args.top, "surprise"))
        show("Board word's WordNet category", *table(ev, lambda c, w: cat_of(w), 30, rng))
        show("Board word's number of WordNet senses (all parts of speech)", *table(
            ev, lambda c, w: n_senses(w), 30, rng))
        show("Clue's category: surprise of the picked word", *table(
            ev, lambda c, w: cat_of(c), 30, rng, sort="surprise"))
        show("Same category as the clue (location and animal clues)", *table(
            ev, lambda c, w: (cat_of(c), "same" if cat_of(w) == cat_of(c) else "other")
            if cat_of(c) in ("location", "animal") else None, 30, rng))


if __name__ == "__main__":
    main()
