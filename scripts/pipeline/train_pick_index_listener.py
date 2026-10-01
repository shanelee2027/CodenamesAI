"""A listener that knows which pick of the turn it is scoring (docs/log.md,
"Pick index as a listener feature").

**Why.** Every booster so far scores a clue's words once, on the full board,
and the same scores serve pick 1, 2, 3 and 4 (only the set of words left
shrinks). The guesser's later picks are measurably different: flatter, and
pulled toward the words already taken. The within-turn model repairs that
with three numbers per pick laid over the scores. Here the booster itself is
told the pick index, so its trees can learn a different shape for each pick.

**The model is still a sequence of conditional choices:** pick x is a
softmax over the words not yet picked, with features (full-board rows, the
incumbent's 44) plus

- `x`: the pick index, 1-based;
- `x_minus_k`: x minus the announced number, so "past the number" is one
  split rather than an interaction the trees must find for each k.

The probability of a whole sequence is still the product of the picks, so
the reward stays an exact subset recursion. In play this costs k booster
runs per clue (one per pick index), not one.

(The marginal framing, "which word is the guesser's x-th choice, among all
the words", was considered and rejected: per-position probabilities cannot
give the probability that picks 1..j are all own words, which is what the
reward needs. With one own word and the assassin, ranked either way half
the time, both positions are own half the time, so independence gives 0.25
where the truth is 0.)

**Arms**, each fitted with listener_training.train's recipe on the train
split of train_listener_net.load_sets, early-stopped on val:
- `depth k`: picks 1..k, the events every booster so far was fitted on;
- `depth 9`: picks 1..min(9, n - 1). gpt-oss ranked the whole board, so
  every stored ranking carries picks beyond the announced number.

Event weights extend the incumbent's: 0.75 per pick into the turn. Every arm
is scored on the same events, picks 1..k, the ones play reads.

Baselines on the same events: a fresh booster on the same 44 features with
the frozen objective (`cache/listener_gbt_control44.txt`, from
train_joint_listener.py), alone, with post-hoc temperatures, and with the
full within-turn model.

    python scripts/pipeline/train_pick_index_listener.py
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
DEPTHS = {"depth k": None, "depth 9": 9}
EXTRA = ["x", "x_minus_k"]


def pick_groups(positions: list[dict], cols: list[int], max_depth: int | None):
    """Choice events with the two pick-index columns appended. With
    `max_depth` None, picks 1..k (build_groups' events, in its order);
    otherwise picks 1..min(max_depth, n - 1)."""
    import codenames.listener_training as T

    X, y, groups, w = [], [], [], []
    for p in positions:
        n, k = p["n"], p["k"]
        depth = min(k if max_depth is None else max_depth, n - 1, len(p["targets"]))
        taken: list[int] = []
        for j in range(depth):
            keep = [r for r in range(n) if r not in taken]
            rows = p["x"][keep][:, cols]
            extra = np.tile([j + 1.0, j + 1.0 - k], (len(keep), 1))
            X.append(np.hstack([rows, extra]))
            lab = np.zeros(len(keep))
            lab[keep.index(p["targets"][j])] = 1.0
            y.append(lab)
            groups.append(len(keep))
            w.append(T.STEP_DECAY ** j)
            taken.append(p["targets"][j])
    return np.vstack(X), np.concatenate(y), groups, np.asarray(w)


def fit(arm: str) -> str:
    import lightgbm as lgb

    import codenames.listener_training as T
    from codenames.listener_features import FEATURE_NAMES
    from train_listener_net import load_sets

    sets = load_sets()
    names = lgb.Booster(model_file=str(CACHE / "listener_gbt.txt")).feature_name()
    cols = [FEATURE_NAMES.index(n) for n in names]
    Xtr, ytr, gtr, wtr = pick_groups(sets["train"], cols, DEPTHS[arm])
    Xva, yva, gva, _ = pick_groups(sets["val"], cols, None)
    print(f"{arm}: {len(gtr)} train events, {len(Xtr)} rows", flush=True)
    b, _ = T.train(Xtr, ytr, gtr, Xva, yva, gva, 8000, 0, names + EXTRA, wtr)
    return b.model_to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    import lightgbm as lgb

    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors
    from codenames.sequential_listener import SequentialParams
    from train_joint_listener import Events
    from train_listener_net import gbt_scores, load_sets

    t0 = time.time()
    with ProcessPoolExecutor(len(DEPTHS)) as ex:
        models = dict(zip(DEPTHS, ex.map(fit, DEPTHS)))
    boosters = {arm: lgb.Booster(model_str=s) for arm, s in models.items()}
    for arm, b in boosters.items():
        path = CACHE / f"listener_gbt_pick_index44_{arm.replace(' ', '')}.txt"
        b.save_model(str(path))
        imp = dict(zip(b.feature_name(), b.feature_importance("gain")))
        tot = sum(imp.values())
        print(f"{arm}: {b.num_trees()} trees -> {path}; gain share x {imp['x'] / tot:.1%}, "
              f"x_minus_k {imp['x_minus_k'] / tot:.1%}")

    sets = load_sets()
    vw = WordVectors()
    control = lgb.Booster(model_file=str(CACHE / "listener_gbt_control44.txt"))
    names = control.feature_name()
    cols = [FEATURE_NAMES.index(n) for n in names]
    load = lambda f: SequentialParams.from_dict(json.loads((CACHE / f).read_text()))
    baselines = [("control, frozen", None), ("control + temperatures", load("sequential_listener_control44_temperature.json")),
                 ("control + within-turn", load("sequential_listener_control44.json"))]
    rng = np.random.default_rng(0)
    for s in ("val", "new boards", "held-out words", "held-out words, Sonnet"):
        e = Events(sets[s], vw)
        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        counts = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (2000, nb))])
        first = e.step == 0
        print(f"\n{s}   ({len(e.step)} events, {first.mean():.0%} at pick 1, {nb} boards)")
        print(f"  {'arm':24s} {'R2':>7s} {'pick 1':>7s} {'picks 2+':>9s}   gain over control + within-turn")
        per_board = {}
        rows = []
        sc = gbt_scores(control, sets[s])
        flat = np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)])
        for label, th in baselines:
            nll, null = e.nll(flat, *e.transform(th, e.drop(sc) if th is not None else None))
            rows.append((label, nll, null))
        X, _, g, _ = pick_groups(sets[s], cols, None)
        assert len(g) == len(e.step)
        for arm, b in boosters.items():
            nll, null = e.nll(b.predict(X, raw_score=True), np.ones(len(e.y)), np.zeros(len(e.y)))
            rows.append((arm, nll, null))
        for label, nll, null in rows:
            r2 = lambda m: 1 - nll[m].mean() / null[m].mean()
            per_board[label] = (np.bincount(bid, nll), np.bincount(bid, null))
            gain = ""
            if label in DEPTHS:
                (a_nll, null_b), (j_nll, _) = per_board["control + within-turn"], per_board[label]
                d = (counts @ (a_nll - j_nll)) / (counts @ null_b)
                lo, hi = np.percentile(d, [2.5, 97.5])
                gain = f"{(a_nll - j_nll).sum() / null_b.sum():+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {label:24s} {r2(np.ones_like(first)):7.4f} {r2(first):7.4f} {r2(~first):9.4f}   {gain}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
