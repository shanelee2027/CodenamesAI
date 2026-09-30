"""Fit and evaluate the within-turn model (codenames/sequential_listener.py)
on the stored rankings' later picks.

**Data.** train_listener_net.py's sets. The listener's scores come from a
saved booster (default: conceptnet_listener's), computed once per position
on the full board, exactly as the reward uses them. The correction's twelve
parameters are fitted on **val**, whose scores are out of sample for the
booster (val only chose its tree count), then scored on new boards, held-out
words (gpt-oss) and held-out words (Sonnet). Twelve parameters against
~15,000 events leaves no room to overfit val.

**Arms**, nested, so each line says what one more term buys:
- frozen: today's reward model (all parameters 0);
- temperature: a per-pick temperature only (a_j), pick_temperature_listener's idea;
- + drop: the temperature also depends on how far the best word left falls
  below the best word already picked (b_j);
- + fit: similarity to the words already picked (beta_j);
- + sense: sharing a picked word's clue sense (gamma_j).

And one arm outside the nesting, **fit only** (beta_j with alpha fixed at 1).
The temperature terms make later picks flatter, so the spymaster claims
fewer words, and gpt-oss punishes every such caution (docs/log.md, role costs
and the outside option). The fit term instead changes WHICH clue wins:
coherent sets gain. `--arm "fit only"` saves that arm alone, to test the two
effects apart in play.

Per pick, the same metrics as every listener table (train_listener_net.metrics
definitions): R², accuracy with ties credited 1/n, and the favourite's mean
predicted probability against how often it is picked.

    python scripts/pipeline/train_sequential_listener.py [--booster cache/listener_gbt_conceptnet.txt]
    python scripts/pipeline/train_sequential_listener.py --arm "fit only" --out cache/sequential_listener_fit_only.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_listener_net import gbt_scores, load_sets  # noqa: E402

from codenames.listener_net import WordVectors  # noqa: E402
from codenames.sequential_listener import N_STEP_PARAMS, SequentialParams, pair_similarity  # noqa: E402

N = 25
ARMS = [("frozen", ()), ("temperature", ("a",)), ("+ drop", ("a", "b")),
        ("+ fit", ("a", "b", "beta")), ("+ sense", ("a", "b", "beta", "gamma")), ("fit only", ("beta",))]
OUT = Path("cache/sequential_listener.json")


def events(ps: list[dict], scores: list[np.ndarray], vw, senses, clue_index) -> dict:
    """Every pick j >= 2 as padded arrays: base scores, words left, max
    similarity to the words picked, same-sense flag, drop, pick index, target."""
    S, L, F, M, D, K, T, J1 = [], [], [], [], [], [], [], []
    for p, s in zip(ps, scores):
        n = p["n"]
        sim = pair_similarity(p["words"], vw)
        ci = clue_index.get(p["clue"].lower())
        if senses is not None and ci is not None:
            cols = np.array([senses.board_pos.get(w.lower(), -1) for w in p["words"]])
            sense = np.nan_to_num(senses.row("sense", ci, cols), nan=-1).astype(int)
        else:
            sense = np.full(n, -1)
        left = np.ones(n, bool)
        for j in range(min(p["k"], n - 1)):
            t = p["targets"][j]
            if j >= 1:
                picked = ~left
                sp = set(sense[picked][sense[picked] >= 0].tolist())
                pad = lambda x, v: np.concatenate([x, np.full(N - n, v)])
                S.append(pad(s, 0.0))
                L.append(pad(left, False))
                F.append(pad(sim[:, picked].max(1), 0.0))
                M.append(pad((np.isin(sense, list(sp)) & (sense >= 0)).astype(float), 0.0))
                D.append(s[left].max() - s[picked].max())
                K.append(min(j + 1, 4) - 2)
                T.append(t)
                J1.append(j + 1)
            left[t] = False
    f = lambda x, dt=torch.float64: torch.tensor(np.array(x), dtype=dt)
    return {"s": f(S), "left": f(L, torch.bool), "fit": f(F), "same": f(M), "drop": f(D),
            "k": f(K, torch.long), "t": f(T, torch.long), "pick": np.array(J1)}


def logp(E: dict, theta: dict) -> torch.Tensor:
    k = E["k"]
    alpha = torch.exp(theta["a"][k] + theta["b"][k] * E["drop"])[:, None]
    z = alpha * E["s"] + theta["beta"][k][:, None] * E["fit"] + theta["gamma"][k][:, None] * E["same"]
    return z.masked_fill(~E["left"], float("-inf")).log_softmax(-1)


def fit(E: dict, free: tuple[str, ...]) -> dict:
    theta = {n: torch.zeros(N_STEP_PARAMS, dtype=torch.float64, requires_grad=n in free)
             for n in ("a", "b", "beta", "gamma")}
    if free:
        opt = torch.optim.LBFGS([theta[n] for n in free], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -logp(E, theta).gather(1, E["t"][:, None]).mean()
            loss.backward()
            return loss

        opt.step(closure)
    return {n: v.detach() for n, v in theta.items()}


def report(E: dict, theta: dict) -> dict:
    lp = logp(E, theta).numpy()
    t = E["t"].numpy()
    idx = np.arange(len(t))
    nll = -lp[idx, t]
    null = np.log(E["left"].numpy().sum(1))
    top = lp.max(1)
    tied = (lp >= top[:, None] - 1e-9).sum(1)
    hit = np.where(lp[idx, t] >= top - 1e-9, 1.0 / tied, 0.0)
    out = {}
    for j in (2, 3, 4):
        m = E["pick"] >= 4 if j == 4 else E["pick"] == j
        out[j] = (1 - nll[m].mean() / null[m].mean(), hit[m].mean(), np.exp(top[m]).mean(), int(m.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", type=Path, default=Path("cache/listener_gbt_conceptnet.txt"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--arm", default="+ fit", choices=[a for a, _ in ARMS], help="the arm to save")
    args = ap.parse_args()

    import lightgbm as lgb

    from codenames.clue_stats import ClueStats
    from codenames.listener_features import ExtraSims
    from codenames.similarity import DEFAULT_CACHE_DIR

    sets = load_sets()
    bst = lgb.Booster(model_file=str(args.booster))
    vw = WordVectors()
    senses = ExtraSims.load(DEFAULT_CACHE_DIR / "wordnet_senses.npz")
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    clue_index = {w.lower(): i for i, w in enumerate(stats.clue_words)}
    E = {name: events(ps, gbt_scores(bst, ps), vw, senses, clue_index)
         for name, ps in sets.items() if name != "train"}

    thetas = {arm: fit(E["val"], free) for arm, free in ARMS}
    for name, ev in E.items():
        print(f"\n{name}   ({len(ev['t'])} events at picks 2+)")
        print(f"  {'arm':12s} " + "   ".join(f"{'pick ' + str(j) + ('+' if j == 4 else ''):>26s}" for j in (2, 3, 4)))
        print(f"  {'':12s} " + "   ".join(f"{'R2':>7s} {'acc':>6s} {'top p':>6s} {'':>4s}" for _ in (2, 3, 4)))
        for arm, _ in ARMS:
            r = report(ev, thetas[arm])
            print(f"  {arm:12s} " + "   ".join(f"{r[j][0]:7.4f} {r[j][1]:6.3f} {r[j][2]:6.3f} {'':>4s}"
                                               for j in (2, 3, 4)))
    # The sense term bought nothing on any set (docs/log.md), so the saved
    # model is the "+ fit" arm by default: gamma stays 0 and play needs no
    # sense table.
    full = SequentialParams(*(thetas[args.arm][n].numpy() for n in ("a", "b", "beta", "gamma")))
    print(f"\nsaved model, the '{args.arm}' arm, fitted on val (picks 2 / 3 / 4+):")
    for n in ("a", "b", "beta", "gamma"):
        print(f"  {n:6s} " + "  ".join(f"{v:+.3f}" for v in getattr(full, n)))
    args.out.write_text(json.dumps({"booster": str(args.booster), **full.to_dict()}, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
