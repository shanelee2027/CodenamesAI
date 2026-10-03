"""Exploratory feature blocks on top of the assoc booster (docs/log.md,
"Exploratory feature blocks").

The listener is assoc_feature_listener's booster (56 features). Each arm adds
one block of columns, computed here from the cached positions rather than in
codenames/listener_features.py, so nothing in the deployed feature pipeline
changes unless a block earns its place:

- **A, board-relative association and ConceptNet.** The embedding features
  come with board-relative versions (gap to the board's best, rank); the
  association and ConceptNet features are absolute only. For `assoc_share`,
  `assoc_rank` and `cn_any`: the word's rank on the board, its share of the
  board's total, and its margin over the best other word. Plus two board
  constants: how many words gpt-oss's lists name, and how many have any
  ConceptNet edge to the clue.
- **B, the clue itself.** Board constants, which a softmax over the board
  cancels, so they can only act through interactions (a vague clue should
  flatten the other features' effect):
  - how vague gpt-oss finds the clue: the share of distinct words across its
    5 association lists, the mean pairwise overlap of the lists, and how
    often they open with the same word;
  - the clue's rarity, concreteness, familiarity, frequency (Brysbaert norms,
    data/norms/) and number of WordNet senses.
- **C, reverse associations.** gpt-oss's free associations *from the board
  word* (scripts/data/collect_associations.py --board): the share of its 5
  lists that name the clue, the mean reciprocal position, and that score's
  rank on the board. SWOW's reverse direction already earned its place.

Every arm is fitted with listener_training.train on train_listener_net.load_sets'
train split, early-stopped on val, and scored like every listener table: R²
pooled, at pick 1 and at picks 2+ (frozen Plackett-Luce), with the gain over
the assoc booster and a 95% bootstrap over boards.

    python scripts/tools/eval_extra_features.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
BASE = CACHE / "listener_gbt_assoc_features.txt"
NORMS = ROOT / "data" / "norms" / "Concreteness_ratings_Brysbaert_et_al_BRM.txt"

A_SRC = ("assoc_share", "assoc_rank", "cn_any")
BLOCKS = {
    "A": [f"a_{s}_{t}" for s in A_SRC for t in ("rank", "share", "margin")] + ["a_n_named", "a_n_cn"],
    "B": ["b_assoc_distinct", "b_assoc_overlap", "b_assoc_first", "b_rarity", "b_conc", "b_known",
          "b_logfreq", "b_senses"],
    "C": ["c_rev_share", "c_rev_rank", "c_rev_board_rank"],
}
ARMS = {"+A": ["A"], "+B": ["B"], "+C": ["C"], "+A+B+C": ["A", "B", "C"]}
SETS = ("val", "new boards", "held-out words, generated", "held-out words", "held-out words, Sonnet")


class Context:
    """Tables the blocks read: association lists by cue, clue norms."""

    def __init__(self):
        from codenames.clue_stats import ClueStats
        from codenames.listener_training import ASSOCIATIONS, _assoc_norm
        from codenames.similarity import DEFAULT_CACHE_DIR

        self.norm = _assoc_norm
        self.lists: dict[str, list[list[str]]] = {}
        conn = sqlite3.connect(f"file:{ASSOCIATIONS}?mode=ro", uri=True)
        for cue, words in conn.execute("SELECT clue, words FROM lists"):
            ws = [self.norm(w) for w in json.loads(words or "[]")]
            if ws:
                self.lists.setdefault(self.norm(cue), []).append(ws)
        conn.close()
        stats = ClueStats.load(DEFAULT_CACHE_DIR)
        self.rarity = {w.lower(): float(r) for w, r in zip(stats.clue_words, stats.rarity_percentile)}
        self.norms: dict[str, tuple[float, float, float]] = {}
        for line in NORMS.read_text().splitlines()[1:]:
            f = line.split("\t")
            try:
                self.norms[f[0].lower()] = (float(f[2]), float(f[6]), np.log10(float(f[7]) + 1.0))
            except (ValueError, IndexError):
                continue
        self._senses: dict[str, float] = {}

    def senses(self, word: str) -> float:
        if word not in self._senses:
            from nltk.corpus import wordnet as wn

            self._senses[word] = float(len(wn.synsets(word.replace(" ", "_"))))
        return self._senses[word]

    def forms(self, word: str) -> set[str]:
        n = self.norm(word)
        return {n, n + "s", n + "es"} | ({n[:-1]} if n.endswith("s") else set())


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """1 for the board's highest value, ties averaged, NaN stays NaN."""
    from scipy.stats import rankdata

    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    if ok.any():
        out[ok] = rankdata(-v[ok], method="average")
    return out


def block_columns(p: dict, block: str, ctx: Context) -> np.ndarray:
    """(n, len(BLOCKS[block])) for one position's full-board rows."""
    from codenames.listener_features import FEATURE_NAMES

    n = p["n"]
    if block == "A":
        cols = []
        for s in A_SRC:
            v = p["x"][:, FEATURE_NAMES.index(s)].astype(np.float64)
            tot = np.nansum(v)
            other_max = np.array([np.nanmax(np.delete(v, i)) if np.isfinite(np.delete(v, i)).any() else np.nan
                                  for i in range(n)])
            cols += [_rank_desc(v), v / tot if tot > 0 else np.full(n, np.nan), v - other_max]
        share = p["x"][:, FEATURE_NAMES.index("assoc_share")]
        cn = p["x"][:, FEATURE_NAMES.index("cn_any")]
        cols += [np.full(n, float(np.nansum(share > 0)) if not np.isnan(share).all() else np.nan),
                 np.full(n, float(np.nansum(cn > 0)) if not np.isnan(cn).all() else np.nan)]
        return np.column_stack(cols)
    clue = p["clue"].lower()
    if block == "B":
        ls = ctx.lists.get(ctx.norm(clue), [])
        if ls:
            sets = [set(x) for x in ls]
            distinct = len(set().union(*sets)) / sum(len(x) for x in ls)
            pairs = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
            overlap = float(np.mean(pairs)) if pairs else np.nan
            firsts = [x[0] for x in ls]
            first = max(firsts.count(f) for f in set(firsts)) / len(firsts)
        else:
            distinct = overlap = first = np.nan
        conc, known, logfreq = ctx.norms.get(clue, (np.nan, np.nan, np.nan))
        row = [distinct, overlap, first, ctx.rarity.get(clue, np.nan), conc, known, logfreq, ctx.senses(clue)]
        return np.tile(row, (n, 1)).astype(np.float64)
    if block == "C":
        target = ctx.forms(clue)
        share, rank = np.full(n, np.nan), np.full(n, np.nan)
        for i, w in enumerate(p["words"]):
            ls = ctx.lists.get(ctx.norm(w), [])
            if not ls:
                continue
            hits = [next((j + 1 for j, x in enumerate(l) if x in target), 0) for l in ls]
            share[i] = np.mean([h > 0 for h in hits])
            rank[i] = np.mean([1.0 / h if h else 0.0 for h in hits])
        return np.column_stack([share, rank, _rank_desc(rank)])
    raise ValueError(block)


def augment(positions: list[dict], blocks: list[str], ctx: Context) -> list[dict]:
    """Positions with the blocks' columns appended to the full-board rows."""
    out = []
    for p in positions:
        q = dict(p)
        q["x"] = np.hstack([p["x"]] + [block_columns(p, b, ctx) for b in blocks])
        out.append(q)
    return out


def arm_columns(blocks: list[str]) -> tuple[list[int], list[str]]:
    """Base column indices plus the appended ones, and their names."""
    import lightgbm as lgb

    from codenames.listener_features import FEATURE_NAMES

    names = lgb.Booster(model_file=str(BASE)).feature_name()
    cols = [FEATURE_NAMES.index(n) for n in names]
    extra = [c for b in blocks for c in BLOCKS[b]]
    return cols + list(range(len(FEATURE_NAMES), len(FEATURE_NAMES) + len(extra))), names + extra


def fit(arm: str) -> str:
    import codenames.listener_training as T
    from train_listener_net import load_sets

    sets = load_sets()
    ctx = Context()
    blocks = ARMS[arm]
    tr, va = augment(sets["train"], blocks, ctx), augment(sets["val"], blocks, ctx)
    cols, names = arm_columns(blocks)
    Xtr, ytr, gtr, _ = T.build_groups(tr)
    Xva, yva, gva, _ = T.build_groups(va)
    print(f"{arm}: {len(names)} features, {len(gtr)} train events", flush=True)
    b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, 0, names, T.step_weights(sets["train"]))
    return b.model_to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()

    with ProcessPoolExecutor(len(args.arms)) as ex:
        models = dict(zip(args.arms, ex.map(fit, args.arms)))

    import lightgbm as lgb

    import codenames.listener_training as T
    from train_listener_net import load_sets

    boosters = {"assoc (base)": lgb.Booster(model_file=str(BASE))}
    for arm, s in models.items():
        b = lgb.Booster(model_str=s)
        b.save_model(str(CACHE / f"listener_gbt_assoc_extra_{arm.strip('+').replace('+', '')}.txt"))
        boosters[arm] = b
        imp = dict(zip(b.feature_name(), b.feature_importance("gain")))
        tot = sum(imp.values())
        new = [c for blk in ARMS[arm] for c in BLOCKS[blk]]
        print(f"{arm}: {b.num_trees()} trees; new features' gain share {sum(imp[c] for c in new) / tot:.1%} ("
              + ", ".join(f"{c} {imp[c] / tot:.1%}" for c in new) + ")")

    sets = load_sets()
    ctx = Context()
    rng = np.random.default_rng(0)
    for s in SETS:
        ps = augment(sets[s], ["A", "B", "C"], ctx)
        X, y, g, seeds = T.build_groups(ps)
        sizes = np.asarray(g)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        first = T.first_step_mask(sets[s])
        _, bid = np.unique(seeds, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        print(f"\n{s}   ({len(g)} events, {nb} boards)")
        print(f"  {'arm':14s} {'R2':>7s} {'pick 1':>7s} {'2+':>7s}   gain over base [95% CI]   pick-1 gain")
        ref = None
        for arm, b in boosters.items():
            blocks = ARMS.get(arm, [])
            cols, _ = arm_columns(["A", "B", "C"])
            n_base = len(cols) - sum(len(v) for v in BLOCKS.values())
            offset = {"A": 0, "B": len(BLOCKS["A"]), "C": len(BLOCKS["A"]) + len(BLOCKS["B"])}
            use = cols[:n_base] + [cols[n_base + offset[k] + i] for k in blocks for i in range(len(BLOCKS[k]))]
            z = b.predict(X[:, use], raw_score=True)
            gmax = np.maximum.reduceat(z, starts)
            lse = np.log(np.add.reduceat(np.exp(z - np.repeat(gmax, sizes)), starts)) + gmax
            nll = lse - np.add.reduceat(z * y, starts)
            null = np.log(sizes)
            r2 = lambda m: 1 - nll[m].mean() / null[m].mean()
            nb_nll, nb_null = np.bincount(bid, nll), np.bincount(bid, null)
            nb1 = np.bincount(bid[first], nll[first], minlength=nb), np.bincount(bid[first], null[first], minlength=nb)
            gain = g1 = ""
            if ref is None:
                ref = (nb_nll, nb_null, nb1)
            else:
                d = (counts @ (ref[0] - nb_nll)) / (counts @ ref[1])
                lo, hi = np.percentile(d, [2.5, 97.5])
                gain = f"{(ref[0] - nb_nll).sum() / ref[1].sum():+.4f} [{lo:+.4f}, {hi:+.4f}]"
                d1 = (counts @ (ref[2][0] - nb1[0])) / (counts @ ref[2][1])
                lo1, hi1 = np.percentile(d1, [2.5, 97.5])
                g1 = f"{(ref[2][0] - nb1[0]).sum() / ref[2][1].sum():+.4f} [{lo1:+.4f}, {hi1:+.4f}]"
            print(f"  {arm:14s} {r2(np.ones_like(first)):7.4f} {r2(first):7.4f} {r2(~first):7.4f}   {gain:26s}  {g1}")


if __name__ == "__main__":
    main()
