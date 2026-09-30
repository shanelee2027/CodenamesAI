"""Which appended feature blocks improve the listener? An ablation on
identical rows (docs/log.md, "ConceptNet and phrase features").

Each arm is a GBT with the incumbent's current recipe (listener_training.train,
fitted on the train split, early-stopped on val, as train_listener_net.py base
does), seeing the 44 original features plus a set of the appended blocks. The
arms run in parallel. Every arm is scored by train_listener_net.metrics on
every test set, so the numbers are directly comparable with the neural
listener's and isa_listener's tables.

Reported per set, all events and pick 1:
- R², accuracy (ties credited 1/n) and top-pick ECE;
- R² on **non-WordNet clues**, the group these blocks were aimed at;
- R² on **touched events**, those where any candidate has a nonzero value in
  the arm's new features. Where no candidate has one, a feature can still
  move scores, but only through tree interactions.

    python scripts/tools/eval_feature_blocks.py [--save-best cache/listener_gbt_conceptnet.txt]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ARMS = {
    "base": [],
    "+is-a": ["taxonomy"],
    "+conceptnet": ["conceptnet", "compound"],
    "+both": ["taxonomy", "conceptnet", "compound"],
}


def names_for(blocks: list[str]) -> list[str]:
    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES

    appended = {"taxonomy", "conceptnet", "compound"}
    drop = {n for b in appended - set(blocks) for n in T.FEATURE_BLOCKS[b]}
    return [n for n in FEATURE_NAMES if n not in drop]


def fit(blocks: list[str]) -> str:
    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES
    from train_listener_net import load_sets

    sets = load_sets()
    names = names_for(blocks)
    cols = [FEATURE_NAMES.index(n) for n in names]
    Xtr, ytr, gtr, _ = T.build_groups(sets["train"])
    Xva, yva, gva, _ = T.build_groups(sets["val"])
    b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, 0, names,
                   T.step_weights(sets["train"]))
    return b.model_to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", nargs=2, action="append", default=[], metavar=("ARM", "PATH"),
                    help="write an arm's booster, e.g. --save +both cache/listener_gbt_x.txt")
    args = ap.parse_args()

    import lightgbm as lgb
    from nltk.corpus import wordnet as wn

    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors
    from train_listener_net import Tensors, event_logp, gbt_scores, load_sets, metrics, pkey

    with ProcessPoolExecutor(len(ARMS)) as ex:
        models = dict(zip(ARMS, ex.map(fit, ARMS.values())))
    boosters = {k: lgb.Booster(model_str=s) for k, s in models.items()}
    for k, b in boosters.items():
        print(f"{k:12s} {b.num_trees()} trees, {b.num_feature()} features")
    for arm, path in args.save:
        boosters[arm].save_model(path)
        print(f"saved {arm} -> {path}")

    sets = load_sets()
    vw = WordVectors()
    F = sets["val"][0]["x"].shape[1]
    new_cols = {arm: [FEATURE_NAMES.index(n) for b in blocks for n in T.FEATURE_BLOCKS[b]]
                for arm, blocks in ARMS.items()}
    in_wn: dict[str, bool] = {}
    for s in [k for k in sets if k != "train"]:
        ps = sets[s]
        per = [min(p["k"], p["n"] - 1) for p in ps]
        for p in ps:
            c = p["clue"].lower()
            if c not in in_wn:
                in_wn[c] = bool(wn.synsets(c))
        non_wn = np.repeat([not in_wn[p["clue"].lower()] for p in ps], per)
        print(f"\n{s}   (non-WordNet clues: {non_wn.mean():.1%} of events)")
        print(f"  {'arm':12s} {'R2':>7s} {'R2 s1':>7s} {'acc':>7s} {'acc s1':>7s} {'ECE top':>8s}"
              f" {'R2 non-WN':>10s} {'R2 touched (share)':>20s}")
        for arm, b in boosters.items():
            base = {pkey(p): sc for p, sc in zip(ps, gbt_scores(b, ps))}
            D = Tensors(ps, vw, np.zeros(F), np.ones(F), base, "cuda")
            lp = event_logp(D)
            m = metrics(lp, D)
            tgt = D.ev_tgt.cpu().numpy()
            nll = -lp[np.arange(len(tgt)), tgt]
            r2 = lambda mask: 1 - nll[mask].mean() / D.null[mask].mean() if mask.any() else float("nan")
            cols = new_cols["+both"]
            touched = np.repeat([bool(np.nan_to_num(p["x"][:, cols]).any()) for p in ps], per)
            print(f"  {arm:12s} {m['r2']:7.4f} {m['r2_step1']:7.4f} {m['acc']:7.4f} {m['acc_step1']:7.4f} "
                  f"{m['ece_top']:8.4f} {r2(non_wn):10.4f} {r2(touched):11.4f} ({touched.mean():.0%})")
    for arm in ("+conceptnet", "+both"):
        imp = dict(zip(boosters[arm].feature_name(), boosters[arm].feature_importance("gain")))
        tot = sum(imp.values())
        print(f"\ngain share in {arm}: " + ", ".join(
            f"{n} {imp[n] / tot:.2%}" for n in FEATURE_NAMES if n in imp and n in
            {x for b in ARMS[arm] for x in T.FEATURE_BLOCKS[b]}))


if __name__ == "__main__":
    main()
