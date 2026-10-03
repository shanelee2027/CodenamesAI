"""A listener that sees the words already picked this turn (docs/log.md,
"History features").

**Why.** The pick-index booster (train_pick_index_listener.py) knows which
pick it is scoring but not which words were taken, and it lost to the
within-turn model, whose one linear term pulls toward words similar to the
picked ones. Here the booster gets the history as features, so the trees can
learn the pull's shape instead of assuming it. This is an offline upper
bound: in play the features depend on the picked set, so the booster would
have to run once per set (about 130 per clue at k <= 4), not once.

**Features**, for each word still left at pick x, from the set P of words
picked before it (missing at pick 1):
- `embeddings` arm: per embedding space, the highest cosine to a word in P
  (5); the mean over spaces of that (the within-turn model's `fit`); the
  mean-space cosine to the last pick alone.
- `all sources` arm adds links between a picked word p and the candidate w,
  each the max over P:
  - SWOW one hop, p cues w and w cues p, and two hops p to w;
  - ConceptNet, any relation either way;
  - gpt-oss's free associations, p names w and w names p.

Both arms are built on the pick-index model at depth 9 (x and x - k as
features, picks up to 9, 0.75-per-pick weights), so the comparison with it
measures the history alone. Same 44 base features, rows, split and recipe as
every arm since train_joint_listener.py. Scored on picks 1..k.

    python scripts/pipeline/train_history_listener.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path(__file__).resolve().parents[2] / "cache"
SPACES = ("glove", "numberbatch", "wikipedia2vec", "glove840", "fasttext")
EMB = [f"hist_cos_{s}" for s in SPACES] + ["hist_fit", "hist_last"]
LINKS = ["hist_swow_fwd", "hist_swow_rev", "hist_swow2_fwd", "hist_cn", "hist_assoc_fwd", "hist_assoc_rev"]
ARMS = {"embeddings": EMB, "all sources": EMB + LINKS}
MAX_DEPTH = 9


class PairTables:
    """Word-to-word links between board words, as dense (400, 400) tables
    indexed by board position: entry [a, b] is a's link to b with a as the
    cue (SWOW), the clue row (ConceptNet, associations). NaN where the source
    has no row for a."""

    def __init__(self):
        import codenames.listener_training as T
        from codenames.clue_stats import ClueStats
        from codenames.listener_features import ExtraSims, SwowTables
        from codenames.similarity import DEFAULT_CACHE_DIR

        swow = SwowTables.load(T.SWOW_TABLES)
        cn = ExtraSims.load(T.CONCEPTNET_SIMS)
        assoc = ExtraSims.load(T.ASSOC_SIMS)
        stats = ClueStats.load(DEFAULT_CACHE_DIR)
        ci = {w.lower(): i for i, w in enumerate(stats.clue_words)}
        self.pos = {w.lower(): i for w, i in swow.board_pos.items()}
        words = sorted(self.pos, key=self.pos.get)
        n = len(words)
        cols_s = np.arange(n)
        self.swow1 = np.full((n, n), np.nan)
        self.swow2 = np.full((n, n), np.nan)
        self.cn = np.full((n, n), np.nan)
        self.assoc = np.full((n, n), np.nan)
        lower = lambda bp: {w.lower(): i for w, i in bp.items()}
        cn_pos, as_pos = lower(cn.board_pos), lower(assoc.board_pos)
        cn_cols = np.array([cn_pos.get(w, -1) for w in words])
        as_cols = np.array([as_pos.get(w, -1) for w in words])
        for a, w in enumerate(words):
            i = ci.get(w)
            if i is None:
                continue
            # A cue with SWOW data: a missing entry means no link (0). A cue
            # without any stays NaN (unknown).
            if swow.one.indptr[i] < swow.one.indptr[i + 1]:
                self.swow1[a] = np.nan_to_num(swow.row(1, i, cols_s))
            if swow.two.indptr[i] < swow.two.indptr[i + 1]:
                self.swow2[a] = np.nan_to_num(swow.row(2, i, cols_s))
            self.cn[a] = cn.row("cn_any", i, cn_cols)
            self.assoc[a] = assoc.row("assoc_share", i, as_cols)

    def rows(self, words: list[str]) -> np.ndarray:
        return np.array([self.pos.get(w.lower(), -1) for w in words])


def _sub(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """table[idx][:, idx] with NaN rows/columns for words not in the table."""
    out = np.full((len(idx), len(idx)), np.nan)
    ok = idx >= 0
    out[np.ix_(ok, ok)] = table[np.ix_(idx[ok], idx[ok])]
    return out


def _nanmax(m: np.ndarray, axis: int) -> np.ndarray:
    allnan = np.isnan(m).all(axis)
    return np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(m), -np.inf, m), axis))


def history_groups(positions: list[dict], cols: list[int], max_depth: int | None, feats: list[str], vw, pt):
    """Choice events (picks 1..k, or 1..min(max_depth, n - 1)) with x,
    x - k and the history features appended."""
    import codenames.listener_training as T

    X, y, groups, w = [], [], [], []
    for p in positions:
        n, k = p["n"], p["k"]
        rows_v = vw.rows(p["words"])
        if rows_v is None:
            cos = np.full((n, n, len(SPACES)), np.nan)
        else:
            v = vw.vecs[rows_v].astype(np.float32)
            ok = vw.ok[rows_v].astype(bool)
            cos = np.einsum("isd,jsd->ijs", v, v).astype(np.float64)
            cos[~(ok[:, None, :] & ok[None, :, :])] = np.nan
        mean_cos = np.nanmean(np.where(np.isnan(cos), np.nan, cos), axis=2) if rows_v is not None else np.full((n, n), np.nan)
        if "hist_swow_fwd" in feats:
            idx = pt.rows(p["words"])
            s1, s2, cnm, am = (_sub(t, idx) for t in (pt.swow1, pt.swow2, pt.cn, pt.assoc))
        depth = min(k if max_depth is None else max_depth, n - 1, len(p["targets"]))
        taken: list[int] = []
        for j in range(depth):
            keep = [r for r in range(n) if r not in taken]
            base = p["x"][keep][:, cols]
            hist = {}
            if taken:
                for si, s in enumerate(SPACES):
                    hist[f"hist_cos_{s}"] = _nanmax(cos[np.ix_(keep, taken)][:, :, si], 1)
                hist["hist_fit"] = _nanmax(mean_cos[np.ix_(keep, taken)], 1)
                hist["hist_last"] = mean_cos[keep, taken[-1]]
                if "hist_swow_fwd" in feats:
                    hist["hist_swow_fwd"] = _nanmax(s1[np.ix_(taken, keep)], 0)
                    hist["hist_swow_rev"] = _nanmax(s1[np.ix_(keep, taken)], 1)
                    hist["hist_swow2_fwd"] = _nanmax(s2[np.ix_(taken, keep)], 0)
                    hist["hist_cn"] = np.fmax(_nanmax(cnm[np.ix_(taken, keep)], 0), _nanmax(cnm[np.ix_(keep, taken)], 1))
                    hist["hist_assoc_fwd"] = _nanmax(am[np.ix_(taken, keep)], 0)
                    hist["hist_assoc_rev"] = _nanmax(am[np.ix_(keep, taken)], 1)
            extra = np.column_stack([np.full(len(keep), j + 1.0), np.full(len(keep), j + 1.0 - k)]
                                    + [hist.get(f, np.full(len(keep), np.nan)) for f in feats])
            X.append(np.hstack([base, extra]))
            lab = np.zeros(len(keep))
            lab[keep.index(p["targets"][j])] = 1.0
            y.append(lab)
            groups.append(len(keep))
            w.append(T.STEP_DECAY ** j)
            taken.append(p["targets"][j])
    return np.vstack(X), np.concatenate(y), groups, np.asarray(w)


def fit(arm: str, features_from: Path = CACHE / "listener_gbt.txt") -> str:
    import lightgbm as lgb

    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors
    from train_listener_net import load_sets

    sets = load_sets()
    vw = WordVectors()
    pt = PairTables() if arm == "all sources" else None
    names = lgb.Booster(model_file=str(features_from)).feature_name()
    cols = [FEATURE_NAMES.index(n) for n in names]
    feats = ARMS[arm]
    Xtr, ytr, gtr, wtr = history_groups(sets["train"], cols, MAX_DEPTH, feats, vw, pt)
    Xva, yva, gva, _ = history_groups(sets["val"], cols, None, feats, vw, pt)
    print(f"{arm}: {len(gtr)} train events, {Xtr.shape} rows", flush=True)
    b, _ = T.train(Xtr, ytr, gtr, Xva, yva, gva, 8000, 0, names + ["x", "x_minus_k"] + feats, wtr)
    return b.model_to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-from", type=Path, default=CACHE / "listener_gbt.txt",
                    help="base features: this booster's columns (default: the incumbent's 44)")
    ap.add_argument("--name", default="44", help="output suffix naming the base features")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--train-only", action="store_true",
                    help="save the boosters and stop (score with scripts/tools/score_turn_models.py)")
    args = ap.parse_args()

    import lightgbm as lgb

    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors
    from codenames.sequential_listener import SequentialParams
    from train_joint_listener import Events
    from train_listener_net import gbt_scores, load_sets
    from train_pick_index_listener import pick_groups

    t0 = time.time()
    with ProcessPoolExecutor(len(args.arms)) as ex:
        models = dict(zip(args.arms, ex.map(fit, args.arms, [args.features_from] * len(args.arms))))
    boosters = {arm: lgb.Booster(model_str=s) for arm, s in models.items()}
    for arm, b in boosters.items():
        path = CACHE / f"listener_gbt_history{args.name}_{arm.replace(' ', '_')}.txt"
        b.save_model(str(path))
        imp = dict(zip(b.feature_name(), b.feature_importance("gain")))
        tot = sum(imp.values())
        print(f"{arm}: {b.num_trees()} trees -> {path}; gain share "
              + ", ".join(f"{f} {imp[f] / tot:.1%}" for f in ["x", "x_minus_k"] + ARMS[arm]))

    if args.train_only:
        return
    sets = load_sets()
    vw = WordVectors()
    pt = PairTables()
    control = lgb.Booster(model_file=str(CACHE / "listener_gbt_control44.txt"))
    pick9 = lgb.Booster(model_file=str(CACHE / "listener_gbt_pick_index44_depth9.txt"))
    cols = [FEATURE_NAMES.index(n) for n in control.feature_name()]
    within = SequentialParams.from_dict(json.loads((CACHE / "sequential_listener_control44.json").read_text()))
    rng = np.random.default_rng(0)
    for s in ("val", "new boards", "held-out words", "held-out words, Sonnet"):
        e = Events(sets[s], vw)
        pick = np.array([j + 1 for p in sets[s] for j in range(min(p["k"], p["n"] - 1))])
        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        sc = gbt_scores(control, sets[s])
        flat = np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)])
        ones, zeros = np.ones(len(e.y)), np.zeros(len(e.y))
        rows = [("control, frozen", *e.nll(flat, ones, zeros)),
                ("control + within-turn", *e.nll(flat, *e.transform(within, e.drop(sc))))]
        Xp, _, _, _ = pick_groups(sets[s], cols, None)
        rows.append(("pick index, depth 9", *e.nll(pick9.predict(Xp, raw_score=True), ones, zeros)))
        for arm, b in boosters.items():
            Xh, _, g, _ = history_groups(sets[s], cols, None, ARMS[arm], vw, pt)
            assert len(g) == len(e.step)
            rows.append((f"history, {arm}", *e.nll(b.predict(Xh, raw_score=True), ones, zeros)))
        print(f"\n{s}   ({len(pick)} events, {nb} boards)")
        print(f"  {'arm':26s} {'R2':>7s}" + "".join(f"{'pick ' + str(j):>8s}" for j in (1, 2, 3, 4))
              + "   gain over within-turn (95% board bootstrap)")
        ref = None
        for label, nll, null in rows:
            r2 = lambda m: 1 - nll[m].mean() / null[m].mean()
            per_b = (np.bincount(bid, nll), np.bincount(bid, null))
            if label == "control + within-turn":
                ref = per_b
            gain = ""
            if label.startswith("history") or label.startswith("pick index"):
                d = (counts @ (ref[0] - per_b[0])) / (counts @ ref[1])
                lo, hi = np.percentile(d, [2.5, 97.5])
                gain = f"{(ref[0] - per_b[0]).sum() / ref[1].sum():+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {label:26s} {r2(pick > 0):7.4f}" + "".join(f"{r2(pick == j):8.4f}" for j in (1, 2, 3, 4))
                  + f"   {gain}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
