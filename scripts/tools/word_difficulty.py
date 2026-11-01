"""Which board words does the listener booster get wrong, and do they share a kind?

Scored on the generated test sets, whose board words no model was fitted on:
`held-out words, generated` (the base game's 150 held-out words) and `online
words, generated` (the online game's 424 words outside the base 400), both
ranked by gpt-oss. Every pick event is a softmax over the words still
unrevealed, so each candidate row has the booster's probability p of being
picked and an outcome y.

**Per word: observed over expected picks.** O is how often the guesser
picked the word, E the sum of p over every event it was a candidate in.
O/E > 1: the booster underrates the word (the guesser goes for it more than
predicted); O/E < 1: it overrates it (false alarms). Across all words
sum O = sum E, since each event has one pick and its p sum to 1, so a group's
O/E says how the booster misallocates between groups, not how sharp it is.
`z` is (O - E) / sqrt(sum p(1-p)), the per-word signal against noise.

**Per word: log-gain when picked.** The mean of log(n p) over the events in
which the word was picked: how much better than uniform the booster saw that
pick coming. Low for words the guesser reaches by routes the features miss.

**Groups are word properties already in cache/**, not hand labels:
WordNet (a proper-name sense; none at all), multi-word names, Brysbaert
concreteness and frequency and WordNet sense count (cache/word_norms.npz),
and whether the word has an entity vector or a fastText vector. Intervals
are 95% bootstraps over positions.

    python scripts/tools/word_difficulty.py
    python scripts/tools/word_difficulty.py --booster listener_gbt.txt --words homer tarzan "st. patrick"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "cache"
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
os.environ.setdefault("NLTK_DATA", str(PROJECT_ROOT / "data" / "nltk_data"))


def events(positions: list[dict], scores: list[np.ndarray], set_name: str):
    """One row per (pick event, candidate): word, y, p, log(n p), position
    index, set."""
    words, y, p, gain, pos = [], [], [], [], []
    for i, (q, s) in enumerate(zip(positions, scores)):
        taken: list[int] = []
        for j in range(min(q["k"], q["n"] - 1)):
            keep = [r for r in range(q["n"]) if r not in taken]
            z = s[keep] - s[keep].max()
            pk = np.exp(z) / np.exp(z).sum()
            words += [q["words"][r].lower() for r in keep]
            y += [r == q["targets"][j] for r in keep]
            p += list(pk)
            gain += list(np.log(len(keep) * pk))
            pos += [i] * len(keep)
            taken.append(q["targets"][j])
    return dict(word=np.array(words), y=np.array(y, float), p=np.array(p), logn_p=np.array(gain),
                pos=np.array(pos), set=np.array([set_name] * len(words)))


def word_properties(words: list[str]) -> dict[str, dict[str, float]]:
    from nltk.corpus import wordnet as wn

    norms = np.load(CACHE / "word_norms.npz")
    nrow = {w.lower(): r for w, r in zip(norms["board_words"], norms["norms"])}
    ent = np.load(CACHE / "entity_sims.npz")
    has_ent = dict(zip((w.lower() for w in ent["board_words"]), np.isfinite(ent["sims"]).any(axis=0)))
    ext = np.load(CACHE / "extra_sims.npz")
    has_ft = dict(zip((w.lower() for w in ext["board_words"]), np.isfinite(ext["fasttext"]).any(axis=0)))
    out = {}
    for w in words:
        syn = wn.synsets(w.replace(" ", "_"))
        r = nrow.get(w, np.full(5, np.nan))
        out[w] = dict(
            proper=float(any(s.instance_hypernyms() for s in syn)),
            proper_first=float(bool(syn) and bool(syn[0].instance_hypernyms())),
            no_wordnet=float(not syn),
            multiword=float(" " in w or "-" in w),
            conc=float(r[0]), log_freq=float(r[3]), senses=float(len(syn)),
            entity=float(bool(has_ent.get(w, False))), fasttext=float(bool(has_ft.get(w, False))),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", default="listener_gbt_assoc_profile.txt", help="file in cache/")
    ap.add_argument("--words", nargs="*", default=["homer", "st. patrick", "tarzan", "snow white", "pegasus"],
                    help="words to show individually")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-picks", type=int, default=5, help="for the per-word lists")
    ap.add_argument("--json", type=Path, default=None, help="also write the per-word table here")
    args = ap.parse_args()

    import lightgbm as lgb

    from eval_listener_accuracy import with_all_columns
    from train_listener_net import GENERATED, GENERATED_ONLINE, gbt_scores, load_sets

    sets = load_sets()
    booster = lgb.Booster(model_file=str(CACHE / args.booster))
    parts, offset = [], 0
    for name in (GENERATED, GENERATED_ONLINE):
        sets[name] = with_all_columns(sets[name])
        e = events(sets[name], gbt_scores(booster, sets[name]), name)
        e["pos"] = e["pos"] + offset
        offset = e["pos"].max() + 1
        parts.append(e)
    ev = {k: np.concatenate([e[k] for e in parts]) for k in parts[0]}
    n_pos = offset

    words = sorted(set(ev["word"]))
    props = word_properties(words)
    widx = {w: i for i, w in enumerate(words)}
    wi = np.array([widx[w] for w in ev["word"]])
    nw = len(words)
    O = np.bincount(wi, ev["y"], nw)
    E = np.bincount(wi, ev["p"], nw)
    V = np.bincount(wi, ev["p"] * (1 - ev["p"]), nw)
    G = np.bincount(wi, ev["y"] * ev["logn_p"], nw)
    C = np.bincount(wi, minlength=nw)
    z = (O - E) / np.sqrt(V)
    print(f"{args.booster}: {len(words)} words, {n_pos} positions, {int(ev['y'].sum())} pick events; "
          f"pooled log-gain when picked {(ev['y'] * ev['logn_p']).sum() / ev['y'].sum():.3f}")

    # Groups, with a bootstrap over positions.
    rng = np.random.default_rng(0)
    draws = rng.integers(0, n_pos, (1000, n_pos))
    counts = np.stack([np.bincount(d, minlength=n_pos) for d in draws])  # (B, n_pos)

    def by_pos(mask: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.bincount(ev["pos"][mask], v[mask], n_pos)

    def group(label: str, row_mask: np.ndarray) -> None:
        o, e = by_pos(row_mask, ev["y"]), by_pos(row_mask, ev["p"])
        g = by_pos(row_mask, ev["y"] * ev["logn_p"])
        ratio = (counts @ o) / (counts @ e)
        gain = (counts @ g) / (counts @ o)
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        glo, ghi = np.percentile(gain, [2.5, 97.5])
        nwords = len(set(ev["word"][row_mask]))
        print(f"  {label:34s} {nwords:4d} words  O/E {o.sum() / e.sum():5.2f} [{lo:.2f}, {hi:.2f}]   "
              f"log-gain {g.sum() / o.sum():6.3f} [{glo:.3f}, {ghi:.3f}]")

    col = {k: np.array([props[w][k] for w in ev["word"]]) for k in props[words[0]]}
    print("\nby test set")
    for name in (GENERATED, GENERATED_ONLINE):
        group(name, ev["set"] == name)
    print("\nby kind of word")
    group("WordNet: a proper-name sense", col["proper"] == 1)
    group("WordNet: first sense a proper name", col["proper_first"] == 1)
    group("not in WordNet", col["no_wordnet"] == 1)
    group("multi-word name", col["multiword"] == 1)
    group("no entity vector", col["entity"] == 0)
    group("no fastText vector", col["fasttext"] == 0)
    group("everything else (none of the above)",
          (col["proper"] == 0) & (col["no_wordnet"] == 0) & (col["multiword"] == 0) & (col["entity"] == 1)
          & (col["fasttext"] == 1))
    for k, label, edges in (("conc", "concreteness (1-5)", None), ("log_freq", "log frequency", None),
                            ("senses", "WordNet senses", [1, 2, 4, 8])):
        print(f"\nby {label}")
        v = col[k]
        if edges is None:
            finite = np.array([props[w][k] for w in words])
            edges = list(np.nanpercentile(finite, [25, 50, 75]))
            bins = [(-np.inf, edges[0]), (edges[0], edges[1]), (edges[1], edges[2]), (edges[2], np.inf)]
            names = [f"Q{i + 1} ({a:.2f} - {b:.2f})" for i, (a, b) in enumerate(bins)]
        else:
            bins = [(-np.inf, 1.5), (1.5, 3.5), (3.5, 7.5), (7.5, np.inf)]
            names = ["1 or none", "2-3", "4-7", "8+"]
        for (a, b), nm in zip(bins, names):
            group(nm, (v > a) & (v <= b) if np.isfinite(a) else v <= b)
        if np.isnan(v).any():
            group("not in the norms", np.isnan(v))

    ok = O >= args.min_picks
    order = np.argsort(z)
    fmt = lambda i: (f"    {words[i]:16s} picked {int(O[i]):3d} expected {E[i]:6.1f}  O/E {O[i] / E[i]:5.2f}  "
                     f"z {z[i]:+5.1f}  log-gain {G[i] / max(O[i], 1):6.3f}  on {C[i]} boards-steps")
    print(f"\nmost overrated (O/E lowest by z, >= {args.min_picks} picks)")
    for i in [i for i in order if ok[i]][: args.top]:
        print(fmt(i))
    print(f"\nmost underrated (O/E highest by z, >= {args.min_picks} picks)")
    for i in [i for i in order[::-1] if ok[i]][: args.top]:
        print(fmt(i))
    print("\nasked for")
    for w in args.words:
        if w.lower() in widx:
            print(fmt(widx[w.lower()]))
        else:
            print(f"    {w}: not on any test board")

    if args.json:
        args.json.write_text(json.dumps({w: dict(O=float(O[i]), E=float(E[i]), z=float(z[i]),
                                                 gain=float(G[i] / max(O[i], 1)), rows=int(C[i]), **props[w])
                                         for w, i in widx.items()}, indent=1))


if __name__ == "__main__":
    main()
