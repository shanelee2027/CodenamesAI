"""Score the distilled listener as a probability model, not as an argmax.

Top-1 accuracy is the wrong headline for this model. `expected_words` never
looks at the listener's favourite word -- it needs P(guess w) for every word on
the board to compute gain and penalty, and a turn where the top word is 95%
likely and a turn where it is 40% likely are completely different turns that
score identically under accuracy.

So the metrics here are the distributional ones: log loss and Brier on the
teacher's actual pick, and calibration of the model's own confidence.

**The incumbent is the Gaussian.** `expected_words` currently assumes
`perceived = z + N(0, sigma)` and derives P(argmax) from it, which is exactly
`listener_features.p_is_max`. That is the thing a distilled listener has to
beat to be worth wiring in, and it gets its best shot here: sigma is fitted on
training boards rather than taken from the config.

**Temperature scaling is fitted on held-out boards.** Fitting it on the GBT's
own training scores picks T<1 -- the training scores are overfit, so the fit
concludes the model should be *sharpened* -- and that makes validation worse.
Measured: T*=0.8 on train scores, 1.7443 val log loss, against 1.7164 at T=1.
Carving a calibration slice out of the training boards gives T*=1.0 instead,
i.e. the group-softmax objective is already calibrated and needs no correction.

Usage:
    python scripts/tools/eval_listener_calibration.py --model <teacher>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.argv = [sys.argv[0]]  # train_listener parses argv at import time otherwise
import scripts.pipeline.train_listener as T  # noqa: E402
from codenames.listener_features import FEATURE_NAMES, p_is_max  # noqa: E402

EPS = 1e-9
BINS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def bounds(groups: list[int]) -> list[tuple[int, int]]:
    b = np.concatenate([[0], np.cumsum(groups)])
    return list(zip(b[:-1], b[1:]))


def targets(y: np.ndarray, B: list[tuple[int, int]]) -> np.ndarray:
    return np.array([int(np.argmax(y[a:b])) for a, b in B])


def softmax_probs(scores: np.ndarray, B, temp: float = 1.0) -> list[np.ndarray]:
    out = []
    for a, b in B:
        s = scores[a:b] / temp
        s = s - s.max()
        e = np.exp(s)
        out.append(e / e.sum())
    return out


def gauss_probs(z: np.ndarray, B, sigma: float) -> list[np.ndarray]:
    """The incumbent, renormalised: quadrature leaves p summing to ~1, not 1."""
    out = []
    for a, b in B:
        p = p_is_max(np.asarray(z[a:b], dtype=np.float64), sigma)
        tot = p.sum()
        out.append(p / tot if tot > 0 else np.full(len(p), 1.0 / len(p)))
    return out


def nll(probs, tgt) -> np.ndarray:
    return -np.log(np.maximum(np.array([p[i] for p, i in zip(probs, tgt)]), EPS))


def brier(probs, tgt) -> np.ndarray:
    return np.array([float(np.sum((p - np.eye(len(p))[i]) ** 2)) for p, i in zip(probs, tgt)])


def conf_hit(probs, tgt) -> tuple[np.ndarray, np.ndarray]:
    c = np.array([p.max() for p in probs])
    h = np.array([float(int(np.argmax(p)) == i) for p, i in zip(probs, tgt)])
    return c, h


def calibration(conf: np.ndarray, hit: np.ndarray, label: str) -> float:
    print(f"\ncalibration of the top choice -- {label}")
    print(f"  {'bin':>14s} {'n':>6s} {'predicted':>10s} {'actual':>8s}")
    ece = 0.0
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 20:
            continue
        ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
        print(f"  [{lo:.2f},{hi:.2f}) {m.sum():6d} {conf[m].mean():10.3f} {hit[m].mean():8.3f}")
    print(f"  ECE {ece:.4f}   top>0.8 on {np.mean(conf > 0.8):.1%} of turns"
          f"   top<0.4 on {np.mean(conf < 0.4):.1%}")
    return ece


def by_context(ng: np.ndarray, nm: np.ndarray, sizes: np.ndarray,
               step: np.ndarray, gname: str) -> None:
    """Break the comparison down by how many words are left and how deep into
    the turn we are.

    These get conflated: a 3-word candidate set means either a mostly-revealed
    BOARD (still the listener's first pick -- the decision the spymaster cares
    about) or a late STEP inside a turn (the teacher has taken its favourites
    and is ranking words it does not want). Raw log loss is not comparable
    across the bins because chance moves with n, so report the gap to uniform:
    how many nats of the available information each model actually captures.
    """
    unif = np.log(sizes.astype(float))
    print("\ninformation captured over uniform (nats) -- higher is better")
    print(f"  {'candidates':>10s} {'step':>9s} {'n':>6s} {'available':>10s} "
          f"{'Gauss':>7s} {'GBT':>7s} {'GBT share':>10s}")
    for lo, hi in [(2, 8), (9, 16), (17, 25)]:
        for label, sel in (("1 (first)", step == 0), ("2+", step > 0)):
            m = (sizes >= lo) & (sizes <= hi) & sel
            if m.sum() < 50:
                continue
            u = unif[m].mean()
            gi, mi = u - ng[m].mean(), u - nm[m].mean()
            print(f"  {lo:2d}-{hi:2d}{'':5s} {label:>9s} {m.sum():6d} {u:10.3f} "
                  f"{gi:7.3f} {mi:7.3f} {mi / u:9.1%}")
    print("\nby step index alone (all board sizes):")
    print(f"  {'step':>5s} {'n':>6s} {'available':>10s} {'Gauss':>7s} {'GBT':>7s}")
    for j in range(5):
        m = step == j
        if m.sum() < 50:
            continue
        u = unif[m].mean()
        print(f"  {j+1:5d} {m.sum():6d} {u:10.3f} {u - ng[m].mean():7.3f} {u - nm[m].mean():7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--model", default="deepinfra/openai/gpt-oss-120b+effort=low")
    ap.add_argument("--max-seed", type=int, default=60)
    ap.add_argument("--collected", type=int, default=40000)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--calib-frac", type=float, default=0.2,
                    help="share of TRAIN boards held out to fit the temperature")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    positions, _ = T.load_positions(args.db, args.model, args.max_seed, args.collected)
    seeds = sorted({p["seed"] for p in positions})
    rng = np.random.default_rng(args.seed)
    val = set(rng.choice(seeds, size=int(len(seeds) * args.val_frac), replace=False).tolist())
    tr_pos = [p for p in positions if p["seed"] not in val]
    va_pos = [p for p in positions if p["seed"] in val]

    tr_seeds = sorted({p["seed"] for p in tr_pos})
    cal = set(np.random.default_rng(args.seed + 7)
              .choice(tr_seeds, size=int(len(tr_seeds) * args.calib_frac), replace=False).tolist())
    fit_pos = [p for p in tr_pos if p["seed"] not in cal]
    cal_pos = [p for p in tr_pos if p["seed"] in cal]

    Xf, yf, gf, _ = T.build_groups(fit_pos)
    Xc, yc, gc, _ = T.build_groups(cal_pos)
    Xv, yv, gv, _ = T.build_groups(va_pos)
    print(f"events: {len(gf)} fit / {len(gc)} calib / {len(gv)} val", flush=True)

    Bc, Bv = bounds(gc), bounds(gv)
    Tc, Tv = targets(yc, Bc), targets(yv, Bv)
    zc, zv = Xc[:, FEATURE_NAMES.index("z_numberbatch")], Xv[:, FEATURE_NAMES.index("z_numberbatch")]

    # Both baselines get their free parameter fitted on the calibration boards.
    soft_T = min((0.25, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0),
                 key=lambda t: nll(softmax_probs(zc, Bc, t), Tc).mean())
    # Fine grid near the optimum on purpose. A coarse one (..., 2.0, 2.5, 3.0)
    # returns 2.5 for gpt-oss, which happens to equal the shipped config value
    # and reads as though the fit recovered it; the actual pooled optimum is
    # 2.4. This sigma describes the TEACHER and has nothing to do with the
    # config -- see docs/log.md.
    sigma = min((1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0, 3.5),
                key=lambda s: nll(gauss_probs(zc, Bc, s), Tc).mean())
    print(f"fitted on held-out boards: softmax T={soft_T}, Gaussian sigma={sigma}", flush=True)

    booster, _ = T.train(Xf, yf, gf, Xc, yc, gc, 3000, args.seed, list(FEATURE_NAMES))
    raw_c, raw_v = booster.predict(Xc, raw_score=True), booster.predict(Xv, raw_score=True)
    gbt_T = min((0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0),
                key=lambda t: nll(softmax_probs(raw_c, Bc, t), Tc).mean())
    print(f"GBT temperature (held-out fit): T={gbt_T}", flush=True)

    models = {
        "uniform": [np.full(b - a, 1.0 / (b - a)) for a, b in Bv],
        f"numberbatch softmax T={soft_T}": softmax_probs(zv, Bv, soft_T),
        f"Gaussian p_is_max sigma={sigma}": gauss_probs(zv, Bv, sigma),
        f"GBT (T={gbt_T})": softmax_probs(raw_v, Bv, gbt_T),
    }
    print(f"\n{'model':32s} {'log loss':>9s} {'Brier':>7s} {'top-1':>7s} {'mean conf':>10s}")
    scored = {}
    for name, probs in models.items():
        n, b = nll(probs, Tv), brier(probs, Tv)
        c, h = conf_hit(probs, Tv)
        scored[name] = (n, c, h)
        print(f"{name:32s} {n.mean():9.4f} {b.mean():7.4f} {h.mean():7.4f} {c.mean():10.4f}")

    gname = f"Gaussian p_is_max sigma={sigma}"
    mname = f"GBT (T={gbt_T})"
    calibration(*scored[gname][1:], gname)
    calibration(*scored[mname][1:], mname)

    d = scored[gname][0] - scored[mname][0]
    idx = np.random.default_rng(args.seed + 1).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    print(f"\nlog-loss improvement, Gaussian -> GBT: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    u = scored["uniform"][0].mean()
    print(f"information captured vs uniform: Gaussian {u - scored[gname][0].mean():.4f} nats, "
          f"GBT {u - scored[mname][0].mean():.4f} nats "
          f"({(u - scored[mname][0].mean()) / (u - scored[gname][0].mean()) - 1:+.1%})")

    step = np.array([j for p in va_pos for j in range(min(p["k"], p["n"] - 1))])
    by_context(scored[gname][0], scored[mname][0], np.array(gv), step, gname)


if __name__ == "__main__":
    main()
