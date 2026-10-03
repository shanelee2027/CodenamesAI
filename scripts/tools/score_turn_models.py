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

Every model is on one base feature set (`--base`: the incumbent's 44, or the
assoc booster's 56) with the same rows, split and recipe, scored on picks 1..k: R² pooled and per pick, and each model's gain
over the within-turn model with a 95% bootstrap over boards.

    python scripts/tools/score_turn_models.py
    python scripts/tools/score_turn_models.py --base assoc
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
# Per base feature set: the frozen booster, its post-hoc temperatures and
# within-turn parameters (fitted on val), and the suffix of the pick-index and
# history boosters trained on the same columns.
BASES = {
    "44": {"booster": "listener_gbt_control44.txt",
           "temperatures": "sequential_listener_control44_temperature.json",
           "within": "sequential_listener_control44.json"},
    "assoc": {"booster": "listener_gbt_assoc_features.txt",
              "temperatures": "sequential_listener_assoc_temperature.json",
              "within": "sequential_listener_assoc.json"},
}
REF = "within-turn"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--base", choices=list(BASES), default="44", help="which base feature set")
    ap.add_argument("--val-temperatures", action="store_true",
                    help="also score every model with per-pick temperatures (picks 1, 2, 3, 4+) fitted on val")
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
    base = BASES[args.base]
    control = lgb.Booster(model_file=str(CACHE / base["booster"]))
    cols = [FEATURE_NAMES.index(n) for n in control.feature_name()]
    seq = lambda f: SequentialParams.from_dict(json.loads((CACHE / f).read_text()))
    rng = np.random.default_rng(0)

    def model_logits(ps, e):
        """(label, preds, alpha, offset) per model; e.nll(preds, alpha, offset)
        is its loss per event."""
        ones, zeros = np.ones(len(e.y)), np.zeros(len(e.y))
        sc = gbt_scores(control, ps)
        flat = np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)])
        out = [("frozen", flat, ones, zeros)]
        for label, f in (("+ temperatures", base["temperatures"]), (REF, base["within"])):
            out.append((label, flat, *e.transform(seq(f), e.drop(sc))))
        Xp, _, _, _ = pick_groups(ps, cols, None)
        for label, f in (("pick index, depth k", f"listener_gbt_pick_index{args.base}_depthk.txt"),
                         ("pick index, depth 9", f"listener_gbt_pick_index{args.base}_depth9.txt")):
            if (CACHE / f).exists():
                out.append((label, lgb.Booster(model_file=str(CACHE / f)).predict(Xp, raw_score=True), ones, zeros))
        for arm, feats in ARMS.items():
            b = lgb.Booster(model_file=str(CACHE / f"listener_gbt_history{args.base}_{arm.replace(' ', '_')}.txt"))
            Xh, _, g, _ = history_groups(ps, cols, None, feats, vw, pt)
            assert len(g) == len(e.step)
            out.append((f"pick index + history, {arm}", b.predict(Xh, raw_score=True), ones, zeros))
        return out

    # Per-pick temperatures fitted on val (gpt-oss, out of sample for every
    # booster, which only chose its tree count there): each model's whole
    # logit at pick j is multiplied by c_j, the maximum-likelihood value on val.
    temps = {}
    if args.val_temperatures:
        from scipy.optimize import minimize_scalar

        ev = Events(sets["val"], vw)
        bucket = np.minimum(ev.step, 3)
        for label, z, al, off in model_logits(sets["val"], ev):
            c = np.ones(4)
            for j in range(4):
                m = bucket == j
                f = lambda t: ev.nll(z, np.exp(t) * al, np.exp(t) * off)[0][m].sum()
                c[j] = np.exp(minimize_scalar(f, bounds=(-2.5, 1.5), method="bounded").x)
            temps[label] = c
            print(f"  val temperatures (1/c) for {label}: " + " / ".join(f"{1 / x:.2f}" for x in c), flush=True)

    for s in args.sets:
        ps = sets[s]
        e = Events(ps, vw)
        pick = np.array([j + 1 for p in ps for j in range(min(p["k"], p["n"] - 1))])
        rows = []
        for label, z, al, off in model_logits(ps, e):
            rows.append((label, *e.nll(z, al, off)))
            if label in temps:
                c = temps[label][np.minimum(e.row_step, 3)]          # per row, as alpha is
                rows.append((f"{label} + val temps", *e.nll(z, c * al, c * off)))

        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        per_b = {label: (np.bincount(bid, nll), np.bincount(bid, null)) for label, nll, null in rows}
        print(f"\n{s}   ({len(pick)} events, {nb} boards; events per pick "
              + " / ".join(str(int((pick == j).sum())) for j in (1, 2, 3, 4)) + ")")
        print(f"  {'model':46s} {'R2':>7s}" + "".join(f"{'pick ' + str(j):>8s}" for j in (1, 2, 3, 4))
              + f"   gain over {REF} [95% CI]")
        for label, nll, null in rows:
            r2 = lambda m: 1 - nll[m].mean() / null[m].mean()
            gain = ""
            if label != REF:
                (rn, rnull), (mn, _) = per_b[REF], per_b[label]
                d = (counts @ (rn - mn)) / (counts @ rnull)
                lo, hi = np.percentile(d, [2.5, 97.5])
                gain = f"{(rn - mn).sum() / rnull.sum():+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {label:46s} {r2(pick > 0):7.4f}" + "".join(f"{r2(pick == j):8.4f}" for j in (1, 2, 3, 4))
                  + f"   {gain}")


if __name__ == "__main__":
    main()
