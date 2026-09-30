"""Do the is-a features (codenames/listener_features.py, "isa") improve the
listener? (docs/log.md, "Is-a features").

Two GBTs with the incumbent's recipe (listener_training.train, fitted on the
train split and early-stopped on val, as train_listener_net.py base does), on
identical rows: every feature, and every feature but the taxonomy block. The
two fits run in parallel. Both are scored by train_listener_net.metrics on
every test set, so the numbers are directly comparable with the neural
listener's table. They are reported on all events, per pick, and on the events
where the features can speak at all: the clue is a WordNet category of at
least one candidate (isa_n >= 1).

    python scripts/tools/eval_isa_features.py [--save cache/listener_gbt_isa_split.txt]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

ARMS = {"without is-a": False, "with is-a": True}


def fit(with_isa: bool) -> str:
    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES
    from train_listener_net import load_sets

    sets = load_sets()
    names = [n for n in FEATURE_NAMES if with_isa or n not in T.FEATURE_BLOCKS["taxonomy"]]
    cols = [FEATURE_NAMES.index(n) for n in names]
    Xtr, ytr, gtr, _ = T.build_groups(sets["train"])
    Xva, yva, gva, _ = T.build_groups(sets["val"])
    b, _ = T.train(Xtr[:, cols], ytr, gtr, Xva[:, cols], yva, gva, 8000, 0, names,
                   T.step_weights(sets["train"]))
    return b.model_to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", type=Path, default=None, help="write the with-is-a booster here")
    args = ap.parse_args()

    import lightgbm as lgb

    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors
    from train_listener_net import Tensors, event_logp, gbt_scores, load_sets, metrics, pkey

    with ProcessPoolExecutor(2) as ex:
        models = dict(zip(ARMS, ex.map(fit, ARMS.values())))
    boosters = {k: lgb.Booster(model_str=s) for k, s in models.items()}
    for k, b in boosters.items():
        print(f"{k}: {b.num_trees()} trees, {b.num_feature()} features")
    if args.save:
        boosters["with is-a"].save_model(str(args.save))

    sets = load_sets()
    vw = WordVectors()
    isa_n = FEATURE_NAMES.index("isa_n")
    F = sets["val"][0]["x"].shape[1]
    for s in [k for k in sets if k != "train"]:
        ps = sets[s]
        print(f"\n{s}")
        print(f"  {'model':14s} {'R2':>7s} {'R2 s1':>7s} {'acc':>7s} {'acc s1':>7s} {'ECE all':>8s} {'ECE top':>8s}"
              f"   R2 by pick 1/2/3/4+            R2 where isa_n>=1 (events)")
        for name, b in boosters.items():
            base = {pkey(p): sc for p, sc in zip(ps, gbt_scores(b, ps))}
            D = Tensors(ps, vw, np.zeros(F), np.ones(F), base, "cuda")
            lp = event_logp(D)
            m = metrics(lp, D)
            tgt = D.ev_tgt.cpu().numpy()
            nll = -lp[np.arange(len(tgt)), tgt]
            step = np.minimum(D.ev_step, 3)
            by = " ".join(f"{1 - nll[step == j].mean() / D.null[step == j].mean():.3f}" for j in range(4)
                          if (step == j).any())
            # isa_n is a board constant: read it off each event's position.
            n_of = np.concatenate([[p["x"][0, isa_n]] * min(p["k"], p["n"] - 1) for p in ps])
            cat = np.nan_to_num(n_of) >= 1
            r2c = 1 - nll[cat].mean() / D.null[cat].mean() if cat.any() else float("nan")
            print(f"  {name:14s} {m['r2']:7.4f} {m['r2_step1']:7.4f} {m['acc']:7.4f} {m['acc_step1']:7.4f} "
                  f"{m['ece_all']:8.4f} {m['ece_top']:8.4f}   {by:30s} {r2c:.4f} ({int(cat.sum())})")
    imp = dict(zip(boosters["with is-a"].feature_name(), boosters["with is-a"].feature_importance("gain")))
    tot = sum(imp.values())
    print("\ngain share of the is-a features: " + ", ".join(f"{n} {imp[n] / tot:.2%}" for n in
                                                           ("isa", "isa_rev", "isa_n")))


if __name__ == "__main__":
    main()
