"""Fit the booster and the within-turn model together, by alternation
(docs/log.md, "Joint refit of the booster and the within-turn model").

**Why.** The booster is trained on every pick of a turn under the frozen
model: pick j is a softmax over the words left with the same scores, at the
same sharpness. The guesser's later picks are flatter than its first and
pulled toward the words already picked (codenames/sequential_listener.py),
so the booster's scores are a compromise: too flat for pick 1, too sharp for
picks 2+. Fitting the within-turn parameters afterwards repairs picks 2+ but
not the compromise baked into the scores. Here the booster is retrained
knowing how later picks will be read:

    pick 1:   z(w) = s(w)
    pick j:   z(w) = alpha_j * s(w) + beta_j * fit(w, picked)
              alpha_j = exp(a_j + b_j * drop)

Pick 1's temperature stays at 1, which pins the scale of s; the alphas are
then how much flatter each later pick is than the first.

**Alternation.**
- Round 0: the control booster, the frozen objective (as every booster so
  far).
- Fit (a, b, beta) on val with the current booster's scores, as
  train_sequential_listener.py does: val scores are out of sample for the
  booster, which only chose its tree count there.
- Retrain the booster from scratch with those parameters held fixed. The
  gradient for a row of a later pick is alpha_j * (p - y), the hessian
  alpha_j^2 * p(1 - p). alpha depends on the scores through `drop`, so it
  is held at the previous booster's value for the whole round.
- Repeat for `--rounds` rounds.

**Held fixed, so only the objective differs from the control:** the
features (the incumbent's 44 by default), the rows and split
(train_listener_net.load_sets), the recipe (listener_training.train's
parameters), and the 0.75-per-pick event weights.

**Report.** McFadden R² on every set, pooled, at pick 1 and at picks 2+, for
the incumbent booster, the control (frozen and with post-hoc within-turn
parameters) and the joint model, with a 95% board bootstrap on the joint
model's gain over the control with post-hoc parameters.

    python scripts/pipeline/train_joint_listener.py --name 44 --rounds 3
    python scripts/pipeline/train_joint_listener.py --name 44 --rounds 3 --arm temperature
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_listener_net import gbt_scores, load_sets  # noqa: E402
from train_sequential_listener import events as seq_events, fit as seq_fit  # noqa: E402

import codenames.listener_training as T  # noqa: E402
from codenames.listener_features import FEATURE_NAMES  # noqa: E402
from codenames.listener_net import WordVectors  # noqa: E402
from codenames.sequential_listener import N_STEP_PARAMS, SequentialParams, pair_similarity  # noqa: E402

CACHE = T.CACHE
ARMS = {"+ fit": ("a", "b", "beta"),   # the deployed within-turn model; the sense term stays 0
        "temperature": ("a",)}         # a per-pick temperature only: no drop slope, no pull


class Events:
    """Every choice event of a set in build_groups order, with what the joint
    objective needs per row: its event's step index (0 for pick 1, else 1/2/3
    for picks 2/3/4+), its fit to the words already picked, and the event's
    `drop` under a given set of scores."""

    def __init__(self, positions: list[dict], vw: WordVectors):
        self.X, self.y, groups, seeds = T.build_groups(positions)
        self.sizes = np.asarray(groups, dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.positions = positions
        self.weights = T.step_weights(positions)
        self.board = seeds
        step, fit, pos_of, keep_of, picked_of = [], [], [], [], []
        for i, p in enumerate(positions):
            n = p["n"]
            sim = pair_similarity(p["words"], vw)
            taken: list[int] = []
            for j in range(min(p["k"], n - 1)):
                keep = [r for r in range(n) if r not in taken]
                step.append(0 if j == 0 else min(j + 1, 4) - 1)
                fit.append(sim[np.ix_(keep, taken)].max(1) if taken else np.zeros(len(keep)))
                pos_of.append(i)
                keep_of.append(keep)
                picked_of.append(list(taken))
                taken.append(p["targets"][j])
        self.step = np.asarray(step)
        self.row_step = np.repeat(self.step, self.sizes)
        self.fit = np.concatenate(fit)
        self.pos_of, self.keep_of, self.picked_of = pos_of, keep_of, picked_of

    def drop(self, scores: list[np.ndarray]) -> np.ndarray:
        """Per event: best score left minus best score picked (0 at pick 1)."""
        out = np.zeros(len(self.step))
        for e, (i, keep, picked) in enumerate(zip(self.pos_of, self.keep_of, self.picked_of)):
            if picked:
                s = scores[i]
                out[e] = s[keep].max() - s[picked].max()
        return out

    def transform(self, theta: SequentialParams | None, drop: np.ndarray | None):
        """Per-row multiplier alpha and offset beta * fit (1 and 0 at pick 1)."""
        if theta is None:
            return np.ones(len(self.y)), np.zeros(len(self.y))
        k = np.maximum(self.step - 1, 0)
        alpha_e = np.where(self.step == 0, 1.0, np.exp(theta.a[k] + theta.b[k] * drop))
        beta_e = np.where(self.step == 0, 0.0, theta.beta[k])
        return np.repeat(alpha_e, self.sizes), np.repeat(beta_e, self.sizes) * self.fit

    def nll(self, preds: np.ndarray, alpha: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per event: negative log-likelihood of the pick, and log(n)."""
        z = alpha * preds + offset
        gmax = np.maximum.reduceat(z, self.starts)
        lse = np.log(np.add.reduceat(np.exp(z - np.repeat(gmax, self.sizes)), self.starts)) + gmax
        zt = np.add.reduceat(z * self.y, self.starts)
        return lse - zt, np.log(self.sizes)


def joint_objective(ev: Events, alpha: np.ndarray, offset: np.ndarray):
    """Group softmax over z = alpha * f + offset, gradients taken in f."""
    row_w = np.repeat(ev.weights, ev.sizes)

    def obj(preds: np.ndarray, dset):
        y = dset.get_label()
        z = alpha * preds + offset
        gmax = np.maximum.reduceat(z, ev.starts)
        e = np.exp(z - np.repeat(gmax, ev.sizes))
        p = e / np.repeat(np.add.reduceat(e, ev.starts), ev.sizes)
        grad = alpha * (p - y) * row_w
        hess = np.maximum(alpha * alpha * p * (1.0 - p), 1e-6) * row_w
        return grad, hess

    return obj


def train_booster(tr: Events, va: Events, cols: list[int], names: list[str], tr_t, va_t, seed: int):
    """listener_training.train's recipe, with the joint objective and the joint
    R² as the stopping metric (the stopping metric is the objective, as there)."""
    import lightgbm as lgb

    params = {
        "objective": joint_objective(tr, *tr_t),
        "learning_rate": 0.01, "num_leaves": 127, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 10.0, "verbosity": -1, "seed": seed, "feature_pre_filter": False,
    }
    dtr = lgb.Dataset(tr.X[:, cols], label=tr.y, feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(va.X[:, cols], label=va.y, feature_name=names, reference=dtr, free_raw_data=False)

    def feval(preds, _dset):
        nll, null = va.nll(preds, *va_t)
        return "r2", 1.0 - nll.mean() / null.mean(), True

    return lgb.train(params, dtr, num_boost_round=8000, valid_sets=[dva], feval=feval,
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def fit_theta(booster, val_positions: list[dict], vw, arm: tuple[str, ...]) -> SequentialParams:
    E = seq_events(val_positions, gbt_scores(booster, val_positions), vw, None, {})
    th = seq_fit(E, arm)
    return SequentialParams(*(th[n].numpy() for n in ("a", "b", "beta", "gamma")))


def scores_by_position(booster, positions):
    return gbt_scores(booster, positions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-from", type=Path, default=CACHE / "listener_gbt.txt",
                    help="take the feature list from this booster (default: the incumbent's 44)")
    ap.add_argument("--name", default="44", help="output suffix: cache/listener_gbt_{control,joint}<name>.txt")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=list(ARMS), default="+ fit",
                    help="which within-turn terms: the deployed model, or the per-pick temperature only")
    args = ap.parse_args()

    import lightgbm as lgb

    t0 = time.time()
    names = lgb.Booster(model_file=str(args.features_from)).feature_name()
    cols = [FEATURE_NAMES.index(n) for n in names]
    sets = load_sets()
    vw = WordVectors()
    ev = {name: Events(ps, vw) for name, ps in sets.items()}
    print(f"{len(names)} features from {args.features_from}; events built ({time.time() - t0:.0f}s)")

    out_control = CACHE / f"listener_gbt_control{args.name}.txt"
    arm = ARMS[args.arm]
    tag = "" if args.arm == "+ fit" else "_temperature"
    out_joint = CACHE / f"listener_gbt_joint{args.name}{tag}.txt"
    seq_control = CACHE / f"sequential_listener_control{args.name}{tag}.json"
    seq_joint = CACHE / f"sequential_listener_joint{args.name}{tag}.json"

    if out_control.exists():
        control = lgb.Booster(model_file=str(out_control))
        print(f"round 0: control loaded from {out_control}, {control.num_trees()} trees")
    else:
        ident = (np.ones(len(ev["train"].y)), np.zeros(len(ev["train"].y)))
        ident_va = (np.ones(len(ev["val"].y)), np.zeros(len(ev["val"].y)))
        control = train_booster(ev["train"], ev["val"], cols, names, ident, ident_va, args.seed)
        control.save_model(str(out_control))
        print(f"round 0: control, {control.num_trees()} trees -> {out_control} ({time.time() - t0:.0f}s)")
    theta_control = fit_theta(control, sets["val"], vw, arm)
    seq_control.write_text(json.dumps({"booster": str(out_control), **theta_control.to_dict()}, indent=1))

    booster, theta, history = control, theta_control, [("control", theta_control)]
    for r in range(1, args.rounds + 1):
        drops = {s: ev[s].drop(scores_by_position(booster, sets[s])) for s in ("train", "val")}
        tr_t = ev["train"].transform(theta, drops["train"])
        va_t = ev["val"].transform(theta, drops["val"])
        booster = train_booster(ev["train"], ev["val"], cols, names, tr_t, va_t, args.seed)
        theta = fit_theta(booster, sets["val"], vw, arm)
        history.append((f"round {r}", theta))
        print(f"round {r}: {booster.num_trees()} trees; theta "
              + "  ".join(f"{n} " + "/".join(f"{v:+.3f}" for v in getattr(theta, n)) for n in arm)
              + f" ({time.time() - t0:.0f}s)")
    booster.save_model(str(out_joint))
    seq_joint.write_text(json.dumps({"booster": str(out_joint), **theta.to_dict()}, indent=1))
    print(f"saved {out_joint}, {seq_joint}")

    print("\nwithin-turn parameters by round (picks 2 / 3 / 4+):")
    for label, th in history:
        print(f"  {label:9s} alpha at drop 0 " + "/".join(f"{np.exp(v):.2f}" for v in th.a)
              + "   b " + "/".join(f"{v:+.3f}" for v in th.b)
              + "   beta " + "/".join(f"{v:.2f}" for v in th.beta))

    incumbent = lgb.Booster(model_file=str(CACHE / "listener_gbt.txt"))
    arms = [("incumbent, frozen", incumbent, None), ("control, frozen", control, None),
            ("control + within-turn", control, theta_control), ("joint", booster, theta)]
    rng = np.random.default_rng(0)
    for s in ("val", "new boards", "held-out words", "held-out words, Sonnet"):
        e = ev[s]
        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        first = e.step == 0
        print(f"\n{s}   ({len(e.step)} events, {first.mean():.0%} at pick 1, {nb} boards)")
        print(f"  {'arm':24s} {'R2':>7s} {'pick 1':>7s} {'picks 2+':>9s}")
        per_board = {}
        for label, b, th in arms:
            sc = scores_by_position(b, sets[s])
            nll, null = e.nll(np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)]),
                              *e.transform(th, e.drop(sc) if th is not None else None))
            r2 = lambda m: 1 - nll[m].mean() / null[m].mean()
            print(f"  {label:24s} {r2(np.ones_like(first)):7.4f} {r2(first):7.4f} {r2(~first):9.4f}")
            per_board[label] = (np.bincount(bid, nll), np.bincount(bid, null))
        (a_nll, null_b), (j_nll, _) = per_board["control + within-turn"], per_board["joint"]
        d = (counts @ (a_nll - j_nll)) / (counts @ null_b)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  joint - (control + within-turn): {(a_nll - j_nll).sum() / null_b.sum():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] (95% board bootstrap)")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
