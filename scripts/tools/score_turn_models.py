"""Score the saved turn models on matched test sets (docs/log.md, "Turn
models on the generated held-out set").

The pick-index and history boosters (train_pick_index_listener.py,
train_history_listener.py) were compared with the within-turn model on
`held-out words`, gpt-oss's rankings from the eval games. Those clues are all
a spymaster's best, so that set is easier than `new boards` and the two say
different things. `held-out words, generated` (collect_listener_data.py
--vocab holdout) is generated exactly like new boards on the held-out
vocabulary, so a gain that holds on new boards but not there is a gain that
does not transfer to new words.

Every model is on the incumbent's 44 features with the same rows, split and
recipe, scored on picks 1..k: R² pooled and per pick, and each model's gain
over the within-turn model with a 95% bootstrap over boards.

    python scripts/tools/score_turn_models.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

CACHE = Path(__file__).resolve().parents[2] / "cache"
SETS = ("new boards", "held-out words, generated")
REF = "within-turn"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    args = ap.parse_args()

    import lightgbm as lgb

    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors
    from codenames.sequential_listener import SequentialParams
    from train_history_listener import ARMS, PairTables, history_groups
    from train_joint_listener import Events
    from train_listener_net import gbt_scores, load_sets
    from train_pick_index_listener import pick_groups

    sets = load_sets()
    vw = WordVectors()
    pt = PairTables()
    control = lgb.Booster(model_file=str(CACHE / "listener_gbt_control44.txt"))
    cols = [FEATURE_NAMES.index(n) for n in control.feature_name()]
    seq = lambda f: SequentialParams.from_dict(json.loads((CACHE / f).read_text()))
    rng = np.random.default_rng(0)
    for s in args.sets:
        ps = sets[s]
        e = Events(ps, vw)
        pick = np.array([j + 1 for p in ps for j in range(min(p["k"], p["n"] - 1))])
        ones, zeros = np.ones(len(e.y)), np.zeros(len(e.y))
        sc = gbt_scores(control, ps)
        flat = np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)])
        rows = [("frozen", *e.nll(flat, ones, zeros))]
        for label, f in (("+ temperatures", "sequential_listener_control44_temperature.json"),
                         (REF, "sequential_listener_control44.json")):
            th = seq(f)
            rows.append((label, *e.nll(flat, *e.transform(th, e.drop(sc)))))
        Xp, _, _, _ = pick_groups(ps, cols, None)
        for label, f in (("pick index, depth k", "listener_gbt_pick_index44_depthk.txt"),
                         ("pick index, depth 9", "listener_gbt_pick_index44_depth9.txt")):
            b = lgb.Booster(model_file=str(CACHE / f))
            rows.append((label, *e.nll(b.predict(Xp, raw_score=True), ones, zeros)))
        for arm, feats in ARMS.items():
            b = lgb.Booster(model_file=str(CACHE / f"listener_gbt_history44_{arm.replace(' ', '_')}.txt"))
            Xh, _, g, _ = history_groups(ps, cols, None, feats, vw, pt)
            assert len(g) == len(e.step)
            rows.append((f"pick index + history, {arm}", *e.nll(b.predict(Xh, raw_score=True), ones, zeros)))

        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        per_b = {label: (np.bincount(bid, nll), np.bincount(bid, null)) for label, nll, null in rows}
        print(f"\n{s}   ({len(pick)} events, {nb} boards; events per pick "
              + " / ".join(str(int((pick == j).sum())) for j in (1, 2, 3, 4)) + ")")
        print(f"  {'model':34s} {'R2':>7s}" + "".join(f"{'pick ' + str(j):>8s}" for j in (1, 2, 3, 4))
              + f"   gain over {REF} [95% CI]")
        for label, nll, null in rows:
            r2 = lambda m: 1 - nll[m].mean() / null[m].mean()
            gain = ""
            if label != REF:
                (rn, rnull), (mn, _) = per_b[REF], per_b[label]
                d = (counts @ (rn - mn)) / (counts @ rnull)
                lo, hi = np.percentile(d, [2.5, 97.5])
                gain = f"{(rn - mn).sum() / rnull.sum():+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {label:34s} {r2(pick > 0):7.4f}" + "".join(f"{r2(pick == j):8.4f}" for j in (1, 2, 3, 4))
                  + f"   {gain}")


if __name__ == "__main__":
    main()
