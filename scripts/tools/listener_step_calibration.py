"""Is the listener as sure of its later picks as it should be?

The expected reward (codenames/pl_reward.py) scores a clue once and, after
each pick, renormalises those same scores over the words left
(Plackett-Luce). So the confidence it has at pick 1 carries into picks 2-4.
If real guessers are less predictable later on, a clue that relies on weak
second, third and fourth words is overvalued.

This measures that, per pick, on rankings no model was fitted on:
- R2, top-1 accuracy, and calibration (ECE over all words and over the top
  word; train_listener_net.metrics) at each pick;
- the top word's mean predicted probability against how often it is picked,
  which says directly whether the model is over- or under-confident there;
- a temperature per pick, fitted on val by maximum likelihood (scores / tau
  then softmax; tau > 1 means the model is too sure), and what it buys on the
  test sets.

Models: the GBT refitted by `train_listener_net.py base` (the incumbent's
recipe on the train split, so val and the test sets are out of sample), scored
the way the reward scores it: frozen scores, renormalised. Optionally saved
networks, which instead re-run on the words left at every pick.

    python scripts/tools/listener_step_calibration.py [--nets resid_no_attention]
    python scripts/tools/listener_step_calibration.py --booster cache/listener_gbt.txt --fit-on "new boards"

`--booster` scores a saved LightGBM listener (e.g. the deployed one) in place
of the refitted GBT. `--fit-on` picks the set the temperatures are fitted on;
use one the booster never trained on.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from train_listener_net import (  # noqa: E402
    BASE_OUT, CACHE, Tensors, event_logp, load_sets,
)

from codenames.listener_net import WordVectors, load_listener_net  # noqa: E402

STEPS = [(0, "pick 1"), (1, "pick 2"), (2, "pick 3"), (3, "pick 4+")]
TAUS = np.round(np.arange(0.5, 4.01, 0.05), 2)
BINS = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])


def step_mask(D: Tensors, j: int) -> np.ndarray:
    return D.ev_step >= j if j == 3 else D.ev_step == j


def tempered(logp: np.ndarray, tau: float) -> np.ndarray:
    z = logp / tau
    m = np.where(np.isfinite(z), z, -np.inf).max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def ece(p: np.ndarray, y: np.ndarray) -> float:
    b = np.digitize(p, BINS) - 1
    return float(sum((b == i).mean() * abs(p[b == i].mean() - y[b == i].mean())
                     for i in range(len(BINS) - 1) if (b == i).any()))


def stats(logp: np.ndarray, D: Tensors, m: np.ndarray) -> dict:
    lp, tgt, av = logp[m], D.ev_tgt.cpu().numpy()[m], D.ev_avail.cpu().numpy()[m]
    E = len(tgt)
    nll = -lp[np.arange(E), tgt]
    top = lp.max(1)
    hit = (lp[np.arange(E), tgt] >= top - 1e-9).astype(float)
    p = np.exp(lp)
    y = np.zeros_like(p)
    y[np.arange(E), tgt] = 1
    return {"n": E, "r2": 1 - nll.mean() / D.null[m].mean(), "nll": nll.mean(), "acc": hit.mean(),
            "top_pred": np.exp(top).mean(), "ece_all": ece(p[av], y[av]), "ece_top": ece(np.exp(top), hit)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nets", nargs="*", default=[])
    ap.add_argument("--booster", type=Path, default=None)
    ap.add_argument("--fit-on", default="val")
    args = ap.parse_args()

    sets = load_sets()
    if args.booster:
        import lightgbm as lgb

        from train_listener_net import gbt_scores, pkey
        bst = lgb.Booster(model_file=str(args.booster))
        base = {pkey(p): sc for ps in sets.values() for p, sc in zip(ps, gbt_scores(bst, ps))}
    else:
        base = pickle.loads(BASE_OUT.read_bytes())
    vw = WordVectors()
    names = [k for k in sets if k != "train" or args.booster]
    models = [(f"GBT {args.booster.name if args.booster else '(refit)'}, frozen scores", None)] + [
        (n, load_listener_net(CACHE / f"listener_net_{n}.pt", vw, "cuda")) for n in args.nets]

    for label, nc in models:
        print(f"\n######## {label}")
        logps, Ds = {}, {}
        for s in names:
            if nc is None:
                F = sets[s][0]["x"].shape[1]
                D = Tensors(sets[s], vw, np.zeros(F), np.ones(F), base, "cuda")
                logps[s] = event_logp(D)
            else:
                net, ck = nc
                D = Tensors(sets[s], vw, ck["mean"], ck["sd"], base, "cuda")
                logps[s] = event_logp(D, net)
            Ds[s] = D
        taus = {}
        for j, _ in STEPS:
            m = step_mask(Ds[args.fit_on], j)
            taus[j] = min(TAUS, key=lambda t: stats(tempered(logps[args.fit_on], t), Ds[args.fit_on], m)["nll"])
        print(f"temperature fitted on {args.fit_on} per pick: "
              + ", ".join(f"{name} {taus[j]:.2f}" for j, name in STEPS))
        for s in names:
            print(f"\n  {s}")
            print(f"    {'':8s} {'events':>7s} {'R2':>7s} {'acc':>6s} {'top pred':>9s} {'ECE all':>8s} "
                  f"{'ECE top':>8s} | with tau: {'R2':>7s} {'top pred':>9s} {'ECE top':>8s}")
            for j, name in STEPS:
                m = step_mask(Ds[s], j)
                if not m.any():
                    continue
                a = stats(logps[s], Ds[s], m)
                b = stats(tempered(logps[s], taus[j]), Ds[s], m)
                print(f"    {name:8s} {a['n']:7d} {a['r2']:7.4f} {a['acc']:6.3f} {a['top_pred']:9.3f} "
                      f"{a['ece_all']:8.4f} {a['ece_top']:8.4f} |           {b['r2']:7.4f} "
                      f"{b['top_pred']:9.3f} {b['ece_top']:8.4f}")


if __name__ == "__main__":
    main()
