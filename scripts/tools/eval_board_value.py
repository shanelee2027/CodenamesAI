"""Does the board predict who wins beyond the score? The go/no-go test for a
board-reading V (method B of win_prob_listener; docs/log.md,
"win_prob_listener: design").

**Data.** The recorded games V(a, b) was built from (scripts/data/
build_win_value.py): 10,266 games of the incumbent against its own variants,
with gpt-oss guessing. Every turn start is one example: the board as it
stood, from the mover's side, labelled by whether the mover went on to win.

**Arms**, each scored out of fold (5 folds grouped by board, since one board
is played many times across the sweeps):
- **count table:** build_win_value's V, rebuilt on the training folds (a
  logistic prior, shrinkage, monotone projection).
- **GBT, counts:** a GBT on (a, b) plus the count table's logit, as a
  sanity check that the GBT itself adds nothing on counts alone.
- **GBT, counts + board:** the same, plus the board features below.

**Board features.** They are cheap by construction, since a V that reads the
board must be evaluated on thousands of after-boards per move. They come from
the clue policy's precomputed clue x word tables (cache/policy_features.npy),
with s(c, w) = the mean z-score over five embedding spaces and only legal
clues counted. Computed for the mover, and again for the opponent from its
own side:
- `best_k` (k = 1..4): max over clues of (the k-th best own word's s) minus
  (the best non-own word's s). How clean the best k-clue on the board is.
- `worst_word`: min over own words of that word's best one-word margin. The
  hardest own word to clue alone.
- `mean_word`: the mean of the same over own words.
- `assassin_pair`: max over own words of max over clues of min(s(c, w),
  s(c, assassin)). How easily an own word and the assassin share a clue.

Reported: out-of-fold log-loss and Brier, with a paired bootstrap over
boards for the board model's gain.

    python scripts/tools/eval_board_value.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from build_win_value import LABEL_PREFIXES, MAX_WORDS, SUITE_GPTOSS, logistic_fit, monotone  # noqa: E402

Z_COLS = slice(0, 5)          # the five embedding z-scores in PolicyFeatures.pair
SIDE_FEATURES = ["best_1", "best_2", "best_3", "best_4", "worst_word", "mean_word", "assassin_pair"]


def load_states(db: Path) -> list[dict]:
    """One dict per turn start: the board's words and roles from team A's
    side, the words revealed so far, the mover, and whether it won."""
    out = []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for label, suite, board, turns, winner in conn.execute(
            "SELECT label, suite_id, board, turns, winner FROM game_records"):
        if not ((label or "").startswith(LABEL_PREFIXES) or suite == SUITE_GPTOSS):
            continue
        if winner not in ("A", "B"):
            continue
        by_role = json.loads(board)
        words = [w for r in ("own", "opponent", "neutral", "assassin") for w in by_role[r]]
        roles = [r for r in ("own", "opponent", "neutral", "assassin") for _ in by_role[r]]
        key = " ".join(sorted(w.lower() for w in words))
        revealed: set[str] = set()
        for t in json.loads(turns):
            out.append({"words": words, "roles": roles, "revealed": frozenset(revealed),
                        "mover": t["team"], "won": float(winner == t["team"]), "board": key})
            revealed |= {w for w, _ in t["guesses"]}
    conn.close()
    return out


def side_features(S, own, bad, ass, legal):
    """The SIDE_FEATURES for one side of one state. `S` is (25, C) on the
    GPU; own/bad/ass are boolean masks over the 25 words, unrevealed only."""
    import torch

    neg = torch.tensor(-1e9, device=S.device)
    Sl = torch.where(legal[None, :], S, neg)
    n_own = int(own.sum())
    out = [np.nan] * len(SIDE_FEATURES)
    if n_own == 0:
        return out
    badmax = Sl[bad].max(0).values if bad.any() else torch.full_like(Sl[0], -1e9)
    top = torch.sort(Sl[own], dim=0, descending=True).values           # (n_own, C)
    for k in range(1, 5):
        if k <= n_own:
            out[k - 1] = float((top[k - 1] - badmax).max())
    # One-word margins: each own word against every other unrevealed word.
    live = own | bad
    per_word = []
    for i in torch.nonzero(own).flatten().tolist():
        others = live.clone()
        others[i] = False
        other_max = Sl[others].max(0).values if others.any() else torch.full_like(Sl[0], -1e9)
        per_word.append(float((Sl[i] - other_max).max()))
    out[4] = min(per_word)
    out[5] = float(np.mean(per_word))
    if ass.any():
        a = int(torch.nonzero(ass).flatten()[0])
        out[6] = float(torch.minimum(Sl[own], Sl[a][None, :]).max())
    return out


def featurize(states: list[dict]) -> np.ndarray:
    """(n_states, 2 + 2 * len(SIDE_FEATURES)): a, b, then the mover's side
    features, then the opponent's."""
    import torch

    from codenames.clue_policy import PolicyFeatures

    pf = PolicyFeatures.load()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache: dict[str, tuple] = {}
    X = np.full((len(states), 2 + 2 * len(SIDE_FEATURES)), np.nan, dtype=np.float32)
    for n, st in enumerate(states):
        if st["board"] not in cache:
            cache.clear()
            idx = np.array([pf.board_index[w.lower()] for w in st["words"]])
            z = np.asarray(pf.pair[idx][:, :, Z_COLS], dtype=np.float32)       # (25, C, 5)
            S = torch.tensor(np.nanmean(np.where(np.isfinite(z), z, np.nan), axis=2), device=dev)
            S = torch.nan_to_num(S, nan=0.0)
            legal = torch.tensor(pf.legal_mask(idx), device=dev)
            cache[st["board"]] = (S, legal)
        S, legal = cache[st["board"]]
        roles = np.array(st["roles"])
        up = torch.tensor([w not in st["revealed"] for w in st["words"]], device=dev)
        mine, theirs = ("own", "opponent") if st["mover"] == "A" else ("opponent", "own")
        m_own = torch.tensor(roles == mine, device=dev) & up
        o_own = torch.tensor(roles == theirs, device=dev) & up
        ass = torch.tensor(roles == "assassin", device=dev) & up
        neu = torch.tensor(roles == "neutral", device=dev) & up
        X[n, 0], X[n, 1] = int(m_own.sum()), int(o_own.sum())
        X[n, 2:2 + len(SIDE_FEATURES)] = side_features(S, m_own, o_own | neu | ass, ass, legal)
        X[n, 2 + len(SIDE_FEATURES):] = side_features(S, o_own, m_own | neu | ass, ass, legal)
        if n % 10000 == 0:
            print(f"  featurized {n}/{len(states)}", flush=True)
    return X


def count_table(a: np.ndarray, b: np.ndarray, y: np.ndarray, prior_weight: float = 20.0) -> np.ndarray:
    n = np.zeros((MAX_WORDS + 1, MAX_WORDS + 1))
    w = np.zeros_like(n)
    np.add.at(n, (a, b), 1)
    np.add.at(w, (a, b), y)
    prior = logistic_fit(n, w)
    V = (w + prior_weight * prior) / (n + prior_weight)
    V[1:, 1:] = monotone(V[1:, 1:], n[1:, 1:] + prior_weight)
    return V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--features-cache", type=Path,
                    default=PROJECT_ROOT / "cache" / "training_data" / "board_value_features.npz")
    args = ap.parse_args()

    import lightgbm as lgb

    states = load_states(args.db)
    y = np.array([s["won"] for s in states])
    boards = np.array([s["board"] for s in states])
    if args.features_cache.exists():
        X = np.load(args.features_cache)["X"]
        assert len(X) == len(states), "stale feature cache; delete it"
    else:
        X = featurize(states)
        np.savez(args.features_cache, X=X)
    a, b = X[:, 0].astype(int), X[:, 1].astype(int)
    names = ["a", "b", "count_logit"] + [f"me_{f}" for f in SIDE_FEATURES] + [f"them_{f}" for f in SIDE_FEATURES]
    print(f"{len(states)} states, {len(set(boards))} boards")

    ub, bid = np.unique(boards, return_inverse=True)
    rng = np.random.default_rng(0)
    fold_of_board = rng.integers(0, args.folds, len(ub))
    fold = fold_of_board[bid]
    pred = {k: np.zeros(len(y)) for k in ("count table", "GBT, counts", "GBT, counts + board")}
    gain_share: dict[str, float] = {}
    for f in range(args.folds):
        tr, te = fold != f, fold == f
        V = count_table(a[tr], b[tr], y[tr])
        pv = np.clip(V[a, b], 1e-4, 1 - 1e-4)
        pred["count table"][te] = pv[te]
        logit = np.log(pv / (1 - pv))
        full = np.column_stack([a, b, logit, X[:, 2:]])
        for arm, cols in (("GBT, counts", [0, 1, 2]), ("GBT, counts + board", list(range(full.shape[1])))):
            # Early-stopping split inside the training folds, by board: the
            # same board recurs across many games, and a by-row split let the
            # trees memorise boards without early stopping noticing.
            inner = (rng.random(len(ub)) < 0.85)[bid[tr]]
            Xtr, ytr = full[tr][:, cols], y[tr]
            dtr = lgb.Dataset(Xtr[inner], ytr[inner], init_score=logit[tr][inner])
            dva = lgb.Dataset(Xtr[~inner], ytr[~inner], init_score=logit[tr][~inner], reference=dtr)
            bst = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 15,
                             "min_data_in_leaf": 200, "feature_fraction": 0.8, "verbose": -1},
                            dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            raw = bst.predict(full[te][:, cols], raw_score=True) + logit[te]
            pred[arm][te] = 1 / (1 + np.exp(-raw))
            if arm == "GBT, counts + board" and f == 0:
                imp = bst.feature_importance("gain")
                gain_share = {names[c]: imp[i] / imp.sum() for i, c in enumerate(cols)}

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    print(f"\n  {'arm':22s} {'log-loss':>9s} {'Brier':>8s}")
    for arm, p in pred.items():
        print(f"  {arm:22s} {ll(p).mean():9.4f} {((p - y) ** 2).mean():8.4f}")
    d = ll(pred["count table"]) - ll(pred["GBT, counts + board"])
    per_board = np.bincount(bid, d)
    n_board = np.bincount(bid)
    draws = rng.integers(0, len(ub), (2000, len(ub)))
    boot = np.array([per_board[dr].sum() / n_board[dr].sum() for dr in draws])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  board model's log-loss gain over the count table: {d.mean():+.4f} per state, "
          f"95% bootstrap over {len(ub)} boards [{lo:+.4f}, {hi:+.4f}]")
    base = ll(np.full(len(y), y.mean())).mean()
    print(f"  (for scale: constant prediction {base:.4f}; share of it the count table explains "
          f"{1 - ll(pred['count table']).mean() / base:.3f}, the board model "
          f"{1 - ll(pred['GBT, counts + board']).mean() / base:.3f})")
    # Decision relevance: within one move, after-boards with the same score
    # differ only in the board term. How far does it move V between states
    # that share (a, b)?
    pc, pb = np.clip(pred["count table"], 1e-6, 1 - 1e-6), np.clip(pred["GBT, counts + board"], 1e-6, 1 - 1e-6)
    shift = pb - pc
    cell = a * 10 + b
    within = np.concatenate([shift[cell == c] - shift[cell == c].mean() for c in np.unique(cell) if (cell == c).sum() > 50])
    print(f"  board term, within a score cell: sd {within.std():.4f} in win probability; "
          f"|shift| > 0.05 for {np.mean(np.abs(within) > 0.05):.1%} of states, > 0.10 for {np.mean(np.abs(within) > 0.10):.1%}")
    print("\n  gain share, fold 0: " + ", ".join(f"{k} {v:.1%}" for k, v in
                                               sorted(gain_share.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
