"""V(a, b): the probability that the side to move wins, given it has `a` own
words left and the other side has `b`, estimated from recorded games.

**What it is for.** win_prob_listener (docs/versions/win_prob_listener.md)
scores a clue by the probability of winning after the turn it produces,
instead of expected words minus hand-set costs. Each way a turn can end
leaves a new (a, b) with the opponent to move, and V says how good that is.

**Whose value.** Every game here was played by the incumbent, or by a
variant of it that differs only in a cost or the outside option, against the
same, with gpt-oss guessing:
- the role-cost sweeps (`role_cost_sweep_v3*`, `rolecost_confirm*`);
- the outside-option sweep (`outside_rec*`);
- the gpt-oss eval suite (holdout_v1_gptoss).

So V is the incumbent's value: the win probability when both sides play like
the incumbent from here on. Choosing clues greedily against it is one step
of policy iteration. That means improving on the incumbent while assuming
later turns are played like the incumbent's.

**Only the score counts.** V does not see which words remain. This is
method A of the design (docs/log.md, "win_prob_listener: design"). It
captures tempo, which the old reward could not see, and nothing about how
hard the leftover words are to clue.

**Smoothing, then monotone.**
- Each cell's rate is shrunk toward a logistic fit on (a, b) with the
  weight of `--prior` games. The data fills the diagonal band densely, but
  corners such as (3, 9) have a handful of games.
- The result is then projected onto tables that fall as the mover's words
  left rise and rise with the other side's. This is weighted 2-D isotonic
  regression by Dykstra's alternating projections, weighted by games plus
  prior.
- Without the projection the raw corners break the order. For example,
  V(9, 2) = 0.16 > V(8, 2) = 0.01: the few games where a side still had 9
  words against 2 were ones it was losing on luck. A spymaster scored
  against such a table could prefer hitting FEWER of its own words, which is
  never right. Both a finding and a limitation: count-only V cannot tell a
  bad board from a bad score.
- V(0, b) = 1: the side to move has no words left, so it has already won.
  This cell is never reached in play, but it keeps the table total.
- V(a, 0) = 0: the other side has already won.

    python scripts/data/build_win_value.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "cache" / "win_value.npz"
LABEL_PREFIXES = ("role_cost_sweep_v3", "rolecost_confirm", "outside_rec")
SUITE_GPTOSS = "93f5a196b26054d1"
MAX_WORDS = 9


def tally(db: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """(games, wins) per (mover's words left, other's words left), counted
    at the start of every turn of every finished game."""
    n = np.zeros((MAX_WORDS + 1, MAX_WORDS + 1))
    w = np.zeros_like(n)
    games = 0
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for label, suite, board, turns, winner in conn.execute(
            "SELECT label, suite_id, board, turns, winner FROM game_records"):
        if not ((label or "").startswith(LABEL_PREFIXES) or suite == SUITE_GPTOSS):
            continue
        if winner not in ("A", "B"):
            continue
        games += 1
        b = json.loads(board)
        left = {"A": len(b["own"]), "B": len(b["opponent"])}
        for t in json.loads(turns):
            me = t["team"]
            other = "B" if me == "A" else "A"
            n[left[me], left[other]] += 1
            w[left[me], left[other]] += winner == me
            # Guess roles are relative to the guessing team.
            for _, role in t["guesses"]:
                if role == "own":
                    left[me] -= 1
                elif role == "opponent":
                    left[other] -= 1
    conn.close()
    return n, w, games


def logistic_fit(n: np.ndarray, w: np.ndarray) -> np.ndarray:
    """A smooth prior: logit V = quadratic in (a, b) plus 1/a and 1/b terms,
    fitted by binomial maximum likelihood on the cells with data."""
    import torch

    a, b = np.meshgrid(np.arange(MAX_WORDS + 1), np.arange(MAX_WORDS + 1), indexing="ij")
    a, b = a[1:, 1:].ravel().astype(float), b[1:, 1:].ravel().astype(float)
    X = np.stack([np.ones_like(a), a, b, a * a, b * b, a * b, 1 / a, 1 / b], 1)
    X = torch.tensor(X / np.abs(X).max(0), dtype=torch.float64)
    N = torch.tensor(n[1:, 1:].ravel(), dtype=torch.float64)
    Wt = torch.tensor(w[1:, 1:].ravel(), dtype=torch.float64)
    theta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ theta
        loss = -(Wt * torch.nn.functional.logsigmoid(z) + (N - Wt) * torch.nn.functional.logsigmoid(-z)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    p = torch.sigmoid(X @ theta).detach().numpy().reshape(MAX_WORDS, MAX_WORDS)
    out = np.zeros((MAX_WORDS + 1, MAX_WORDS + 1))
    out[1:, 1:] = p
    return out


def monotone(V: np.ndarray, weight: np.ndarray, iters: int = 500) -> np.ndarray:
    """Weighted least-squares projection of V (a x b) onto tables that are
    non-increasing in a and non-decreasing in b: Dykstra's algorithm over the
    two cones, each projected by per-row / per-column isotonic regression."""
    from sklearn.isotonic import IsotonicRegression

    def rows_up(X):          # each row non-decreasing in b
        return np.stack([IsotonicRegression(increasing=True).fit_transform(np.arange(X.shape[1]), x, sample_weight=wt)
                         for x, wt in zip(X, weight)])

    def cols_down(X):        # each column non-increasing in a
        return np.stack([IsotonicRegression(increasing=False).fit_transform(np.arange(X.shape[0]), x, sample_weight=wt)
                         for x, wt in zip(X.T, weight.T)], axis=1)

    x, p, q = V.copy(), np.zeros_like(V), np.zeros_like(V)
    for _ in range(iters):
        y = rows_up(x + p)
        p = x + p - y
        x_new = cols_down(y + q)
        q = y + q - x_new
        done = np.abs(x_new - x).max() < 1e-10
        x = x_new
        if done:
            break
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--prior", type=float, default=20.0, help="games' weight of the logistic prior per cell")
    args = ap.parse_args()

    n, w, games = tally(args.db)
    prior = logistic_fit(n, w)
    V = (w + args.prior * prior) / (n + args.prior)
    raw_inner = V[1:, 1:].copy()
    V[1:, 1:] = monotone(raw_inner, n[1:, 1:] + args.prior)
    moved = np.abs(V[1:, 1:] - raw_inner)
    print(f"monotone projection: largest change {moved.max():.3f} at (a, b) = "
          f"{tuple(int(i) + 1 for i in np.unravel_index(moved.argmax(), moved.shape))}; "
          f"game-weighted mean change {(moved * n[1:, 1:]).sum() / n.sum():.4f}")
    V[0, :] = 1.0
    V[:, 0] = 0.0
    V[0, 0] = np.nan          # both sides out at once cannot happen

    print(f"{games} games, {int(n.sum())} turn states")
    print("V(a, b), a = mover's words left (rows), b = the other side's (columns)")
    print("     " + " ".join(f"{b:5d}" for b in range(1, MAX_WORDS + 1)))
    for a in range(1, MAX_WORDS + 1):
        print(f"  {a}  " + " ".join(f"{V[a, b]:5.2f}" for b in range(1, MAX_WORDS + 1)))
    inner = V[1:, 1:]
    bad_a = int((np.diff(inner, axis=0) > 1e-9).sum())
    bad_b = int((np.diff(inner, axis=1) < -1e-9).sum())
    print(f"monotonicity violations: {bad_a} rising in own words left, {bad_b} falling in the other's")
    np.savez(args.out, V=V, n=n, wins=w, prior=prior, unprojected=np.pad(raw_inner, ((1, 0), (1, 0))))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
