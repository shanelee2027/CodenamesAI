"""A board-reading value model trained on simulated games, and whether it
transfers to real ones (docs/log.md, "Simulated games").

**Data.**
- Simulated: games in cache/sim_games.db under `--label`
  (scripts/pipeline/simulate_games.py), with a fitted listener as the
  guesser. Every turn start is one state, labelled by whether the mover won.
- Real: the recorded gpt-oss games of scripts/tools/eval_board_value.py
  (10,266 games of the incumbent against its variants), the same states and
  features as that go/no-go test.

Features are eval_board_value's: the score (a, b) plus cheap board summaries
for each side (how clean the best k-clue is, the hardest own word, assassin
closeness).

**Reported.**
1. In simulation, out of fold by board: the count table against a GBT on
   counts + board, as in the go/no-go, now with far more games.
2. Transfer: models fitted on all simulated states, scored on the real
   states, against the real count table fitted out of fold on real games.
   - the simulated count table alone (does simulation get the score values
     right?);
   - the real count table plus the simulated board term (does what the board
     is worth in simulation hold for gpt-oss?). This is the quantity a
     board-reading V adds within a move: after-boards of one move differ
     in the board term.

The final simulated board model is saved to cache/board_value_sim.txt, and
the simulated count table its `count_logit` input is read from to
cache/board_value_sim_counts.npz (spymasters/board_value_listener.py uses
both).

    python scripts/tools/eval_sim_value.py --label sim_v1
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from eval_board_value import SIDE_FEATURES, count_table, featurize, load_states  # noqa: E402

SIM_DB = PROJECT_ROOT / "cache" / "sim_games.db"
REAL_FEATURES = PROJECT_ROOT / "cache" / "training_data" / "board_value_features.npz"
OUT = PROJECT_ROOT / "cache" / "board_value_sim.txt"
OUT_COUNTS = PROJECT_ROOT / "cache" / "board_value_sim_counts.npz"     # the count table its input logit is read from
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "verbose": -1}


def load_sim_states(db: Path, label: str, mover: str | None = None) -> list[dict]:
    """Turn starts of the games under `label`; with `mover`, only those where
    the spymaster whose name starts with it is about to move."""
    out = []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for run, board, turns, winner in conn.execute(
            "SELECT label, board, turns, winner FROM game_records WHERE label LIKE ?", (label + "|%",)):
        if winner not in ("A", "B"):
            continue
        seat = dict(kv.split("=", 1) for kv in run.split("|", 1)[1].split(","))
        by_role = json.loads(board)
        words = [w for r in ("own", "opponent", "neutral", "assassin") for w in by_role[r]]
        roles = [r for r in ("own", "opponent", "neutral", "assassin") for _ in by_role[r]]
        key = " ".join(sorted(w.lower() for w in words))
        revealed: set[str] = set()
        for t in json.loads(turns):
            if mover is None or seat[t["team"]].startswith(mover):
                out.append({"words": words, "roles": roles, "revealed": frozenset(revealed),
                            "mover": t["team"], "won": float(winner == t["team"]), "board": key})
            revealed |= {w for w, _ in t["guesses"]}
    conn.close()
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_board(X, y, base_logit, bid, rng):
    """GBT on counts + board with the count logit as the starting score;
    early stopping on a held-out 15% of boards."""
    import lightgbm as lgb

    full = np.column_stack([X[:, :2], base_logit, X[:, 2:]])
    inner = (rng.random(bid.max() + 1) < 0.85)[bid]
    dtr = lgb.Dataset(full[inner], y[inner], init_score=base_logit[inner])
    dva = lgb.Dataset(full[~inner], y[~inner], init_score=base_logit[~inner], reference=dtr)
    return lgb.train(PARAMS, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def board_term(bst, X, base_logit):
    """The trees' shift of the logit away from the count table's. LightGBM's
    raw prediction leaves out the init score it was trained from, so this is
    the board's own contribution; the full logit is base_logit + this."""
    full = np.column_stack([X[:, :2], base_logit, X[:, 2:]])
    return bst.predict(full, raw_score=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--db", type=Path, default=SIM_DB)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mover", default=None,
                    help="only states where this spymaster moves (win_prob_listener reads V at the "
                         "opponent's turns, so `learned_listener` is the table it would use)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    sim = load_sim_states(args.db, args.label, args.mover)
    ys = np.array([s["won"] for s in sim])
    cache = PROJECT_ROOT / "cache" / "training_data" / f"sim_value_features_{args.label}_{args.mover or 'all'}.npz"
    if cache.exists() and len(np.load(cache)["X"]) == len(sim):
        Xs = np.load(cache)["X"]
    else:
        Xs = featurize(sim)
        np.savez(cache, X=Xs)
    ub, bid = np.unique([s["board"] for s in sim], return_inverse=True)
    a, b = Xs[:, 0].astype(int), Xs[:, 1].astype(int)
    print(f"simulated: {len(sim)} states from {len(ub)} boards; mover win rate {ys.mean():.3f}")

    # 1. In simulation, out of fold.
    fold = rng.integers(0, args.folds, len(ub))[bid]
    p_count, p_board = np.zeros(len(ys)), np.zeros(len(ys))
    for f in range(args.folds):
        tr, te = fold != f, fold == f
        V = count_table(a[tr], b[tr], ys[tr])
        lg = logit(V[a, b])
        _, bid_tr = np.unique(bid[tr], return_inverse=True)
        bst = fit_board(Xs[tr], ys[tr], lg[tr], bid_tr, rng)
        p_count[te] = 1 / (1 + np.exp(-lg[te]))
        p_board[te] = 1 / (1 + np.exp(-(lg[te] + board_term(bst, Xs[te], lg[te]))))
    d = ll(p_count, ys) - ll(p_board, ys)
    per_b, n_b = np.bincount(bid, d), np.bincount(bid)
    boot = [per_b[dr].sum() / n_b[dr].sum() for dr in rng.integers(0, len(ub), (2000, len(ub)))]
    print(f"\n1. simulation, out of fold: log-loss count {ll(p_count, ys).mean():.4f}, "
          f"count + board {ll(p_board, ys).mean():.4f}; gain {d.mean():+.4f} "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
    cell = a * 10 + b
    shift = p_board - p_count
    within = np.concatenate([shift[cell == c] - shift[cell == c].mean() for c in np.unique(cell) if (cell == c).sum() > 50])
    print(f"   board term within a score cell: sd {within.std():.4f}; |shift| > 0.05 for "
          f"{np.mean(np.abs(within) > 0.05):.1%}, > 0.10 for {np.mean(np.abs(within) > 0.10):.1%}")

    # Final simulated models on all states.
    V_sim = count_table(a, b, ys)
    lg_all = logit(V_sim[a, b])
    bst = fit_board(Xs, ys, lg_all, bid, rng)
    bst.save_model(str(OUT))
    np.savez(OUT_COUNTS, V=V_sim)
    imp = bst.feature_importance("gain")
    names = ["a", "b", "count_logit"] + [f"me_{f}" for f in SIDE_FEATURES] + [f"them_{f}" for f in SIDE_FEATURES]
    print("   gain share: " + ", ".join(f"{n} {g / imp.sum():.1%}" for n, g in
                                        sorted(zip(names, imp), key=lambda t: -t[1])[:8]))

    # 2. Transfer to real games.
    real = load_states(PROJECT_ROOT / "cache" / "llm_store.db")
    yr = np.array([s["won"] for s in real])
    Xr = np.load(REAL_FEATURES)["X"] if REAL_FEATURES.exists() else None
    if Xr is None or len(Xr) != len(real):          # more games recorded since: rebuild
        Xr = featurize(real)
        np.savez(REAL_FEATURES, X=Xr)
    ar, br = Xr[:, 0].astype(int), Xr[:, 1].astype(int)
    ubr, bidr = np.unique([s["board"] for s in real], return_inverse=True)
    foldr = rng.integers(0, args.folds, len(ubr))[bidr]
    lg_real = np.zeros(len(yr))
    for f in range(args.folds):
        tr, te = foldr != f, foldr == f
        lg_real[te] = logit(count_table(ar[tr], br[tr], yr[tr])[ar[te], br[te]])
    sim_count_lg = logit(V_sim[ar, br])
    term = board_term(bst, Xr, sim_count_lg)
    arms = {"real count table (out of fold)": lg_real,
            "simulated count table": sim_count_lg,
            "real count + simulated board term": lg_real + term}
    print(f"\n2. transfer to {len(real)} real states ({len(ubr)} boards; mover win rate {yr.mean():.3f})")
    base = ll(1 / (1 + np.exp(-lg_real)), yr)
    for name, lgx in arms.items():
        p = 1 / (1 + np.exp(-lgx))
        dd = base - ll(p, yr)
        pb, nb = np.bincount(bidr, dd), np.bincount(bidr)
        bt = [pb[dr].sum() / nb[dr].sum() for dr in rng.integers(0, len(ubr), (2000, len(ubr)))]
        print(f"   {name:36s} log-loss {ll(p, yr).mean():.4f}  gain over real count {dd.mean():+.4f} "
              f"[{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}]")
    # Calibration of the simulated count table on real states, by predicted bin.
    p = 1 / (1 + np.exp(-sim_count_lg))
    print("   simulated count table on real states, predicted / actual by bin:")
    for lo in np.arange(0, 1, 0.2):
        m = (p >= lo) & (p < lo + 0.2)
        if m.sum() > 100:
            print(f"     [{lo:.1f}, {lo + 0.2:.1f})  n={m.sum():6d}  {p[m].mean():.3f} / {yr[m].mean():.3f}")


if __name__ == "__main__":
    main()
