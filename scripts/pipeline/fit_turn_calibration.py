"""Calibrate the listener's turn predictions on the turns a spymaster actually
played (docs/log.md, "win_prob_listener: turn calibration").

**Why.** win_prob_listener uses the listener's probabilities directly. On
the clues it chose they were optimistic (scripts/tools/check_turn_
calibration.py):
- own words over-predicted by about 0.13 per turn;
- opponent endings under-predicted by about 20%;
- the assassin under-predicted where the search likes to go (3-6%
  predicted risk: 10 hits against 5.0 expected).

Much of this is selection, the optimiser's curse. The search picks the clues
whose listener errors flatter them, so on the chosen clues the listener
rates own words too high and bad words too low. A correction must therefore
be fitted on turns that the calibrated policy itself chose. Turns chosen by
another policy carry another selection.

**Model.** The listener score s of each unrevealed word becomes

    s' = alpha * s + beta[role]        (beta[neutral] = 0)

- alpha is one temperature.
- The betas shift each role. The guesser cannot see roles; the betas stand
  in for the search's role-aligned selection bias.

**Fit.** By maximum likelihood of the guesser's actual pick sequence on
each turn, under Plackett-Luce over the unrevealed words. Two arms:
- alpha only;
- alpha + betas.

Each is scored out of fold, 5 folds grouped by board, with the same
predicted-vs-actual table as check_turn_calibration.py. The chosen arm is
refitted on all turns and saved.

**Data.** Games recorded under `--label` (training boards, never the eval
suite's). Only the turns of the spymaster on `--giver`'s seat are used.
Scores come from that spymaster's own booster, `--booster`, read through
`listen`, exactly as its search read them.

    python scripts/pipeline/fit_turn_calibration.py --label calib_winprob_assoc_v1 \\
        --giver win_prob_listener --booster cache/listener_gbt_assoc_features.txt \\
        --out cache/turn_calibration_assoc.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("own", "neutral", "opponent", "assassin")
ARMS = {"alpha": ("alpha",), "alpha + role offsets": ("alpha", "beta")}


def load_turns(db: Path, label: str, giver: str, booster: Path) -> list[dict]:
    """Every turn the giver's seat played under `label`: listener scores for
    the unrevealed words, their roles from the giver's side, and the picks."""
    from codenames.board import Board, Card, OpponentBoardView
    from codenames.game import Role
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    cls, kw = spymaster_spec(giver)
    sm = cls(**{**kw, "model_path": booster})
    out = []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for row_label, board_json, turns_json, seed in conn.execute(
            "SELECT label, board, turns, seed FROM game_records WHERE label LIKE ?", (label + "%",)):
        # Labels end "|A=<name[params]>,B=<name[params]>", and params hold
        # commas of their own, so split on ",B=" only.
        seat_a, seat_b = row_label.split("|", 1)[1].split(",B=", 1)
        seats = {"A": seat_a.removeprefix("A="), "B": seat_b}
        mine = [t for t, name in seats.items() if name.startswith(giver)]
        if len(mine) != 1:
            continue
        team = mine[0]
        by_role = json.loads(board_json)
        cards = tuple(Card(w, Role(r)) for r, ws in by_role.items() for w in ws)
        board = Board(cards=cards, seed=0)
        key = " ".join(sorted(w.lower() for ws in by_role.values() for w in ws))
        for t in json.loads(turns_json):
            if t["team"] == team:
                view = board if team == "A" else OpponentBoardView(board)
                r = sm.listen(view, t["clue"], sims)
                if r is not None and t["guesses"]:
                    words = [w.lower() for w in r["words"]]
                    picks = [words.index(w.lower()) for w, _ in t["guesses"]]
                    out.append({"s": np.asarray(r["scores"], dtype=np.float64),
                                "role": np.array([ROLES.index(x.value) for x in r["roles"]]),
                                "picks": picks, "k": t["number"], "board": key})
            for w, _ in t["guesses"]:
                board.reveal(w)
    conn.close()
    return out


def tensors(turns: list[dict]):
    """Every pick as one padded softmax event: scores, roles, the words still
    available, and the index picked."""
    import torch

    S, R, L, T = [], [], [], []
    for tr in turns:
        n = len(tr["s"])
        left = np.ones(n, bool)
        for p in tr["picks"]:
            pad = 25 - n
            S.append(np.concatenate([tr["s"], np.zeros(pad)]))
            R.append(np.concatenate([tr["role"], np.ones(pad, int)]))
            L.append(np.concatenate([left, np.zeros(pad, bool)]))
            T.append(p)
            left = left.copy()
            left[p] = False
    return (torch.tensor(np.array(S)), torch.tensor(np.array(R)), torch.tensor(np.array(L)),
            torch.tensor(np.array(T)))


def nll(theta, S, R, L, T):
    import torch

    beta = torch.cat([theta["beta"][:1], torch.zeros(1, dtype=S.dtype), theta["beta"][1:]])   # neutral = 0
    z = theta["alpha"] * S + beta[R]
    z = z.masked_fill(~L, float("-inf"))
    return -(z.log_softmax(-1).gather(1, T[:, None])).sum()


def fit(turns: list[dict], free: tuple[str, ...]) -> dict:
    import torch

    S, R, L, T = tensors(turns)
    theta = {"alpha": torch.tensor(1.0, dtype=torch.float64, requires_grad="alpha" in free),
             "beta": torch.zeros(3, dtype=torch.float64, requires_grad="beta" in free)}   # own, opponent, assassin
    params = [theta[n] for n in free]
    opt = torch.optim.LBFGS(params, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(theta, S, R, L, T) / len(T)
        loss.backward()
        return loss

    opt.step(closure)
    return {"alpha": theta["alpha"].item(), "beta_own": theta["beta"][0].item(),
            "beta_opponent": theta["beta"][1].item(), "beta_assassin": theta["beta"][2].item()}


def apply(c: dict, s: np.ndarray, role: np.ndarray) -> np.ndarray:
    beta = np.array([c["beta_own"], 0.0, c["beta_opponent"], c["beta_assassin"]])
    return c["alpha"] * s + beta[role]


def turn_predictions(tr: dict, s: np.ndarray) -> tuple[float, float, float]:
    """(P(assassin ends the turn), P(an opponent word ends it), expected own
    words) at the announced number, from the exact subset DP."""
    from codenames.sequential_listener import SequentialParams, sequential_gain_and_penalty

    own = tr["role"] == 0
    bad_roles = tr["role"][~own]
    k = min(tr["k"], int(own.sum()))
    s_own, s_bad = s[own][None, :], s[~own][None, :]
    zero = np.zeros((25, 25))
    out = []
    for r in (3, 2):
        g, p = sequential_gain_and_penalty(s_own, s_bad, (bad_roles == r).astype(float), k, zero,
                                           SequentialParams.identity())
        out.append(float(p[0, k - 1]))
    return out[0], out[1], float(g[0, k - 1])


def realised(tr: dict) -> tuple[float, float, float]:
    roles = [tr["role"][p] for p in tr["picks"]]
    return float(3 in roles), float(roles[-1] == 2), float(sum(r == 0 for r in roles))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--label", required=True)
    ap.add_argument("--giver", default="win_prob_listener")
    ap.add_argument("--booster", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    turns = load_turns(args.db, args.label, args.giver, args.booster)
    boards = sorted({t["board"] for t in turns})
    rng = np.random.default_rng(0)
    fold_of = dict(zip(boards, rng.integers(0, args.folds, len(boards))))
    print(f"{len(turns)} turns on {len(boards)} boards; assassin ended "
          f"{sum(realised(t)[0] for t in turns):.0f}")

    oof = {arm: [None] * len(turns) for arm in ["uncalibrated", *ARMS]}
    ll = {arm: 0.0 for arm in oof}
    import torch

    for f in range(args.folds):
        tr = [t for t in turns if fold_of[t["board"]] != f]
        te_idx = [i for i, t in enumerate(turns) if fold_of[t["board"]] == f]
        cal = {"uncalibrated": {"alpha": 1.0, "beta_own": 0.0, "beta_opponent": 0.0, "beta_assassin": 0.0}}
        cal.update({arm: fit(tr, free) for arm, free in ARMS.items()})
        for arm, c in cal.items():
            S, R, L, T = tensors([turns[i] for i in te_idx])
            th = {"alpha": torch.tensor(c["alpha"], dtype=torch.float64),
                  "beta": torch.tensor([c["beta_own"], c["beta_opponent"], c["beta_assassin"]], dtype=torch.float64)}
            ll[arm] += float(nll(th, S, R, L, T))
            for i in te_idx:
                oof[arm][i] = turn_predictions(turns[i], apply(c, turns[i]["s"], turns[i]["role"]))
    n_events = sum(len(t["picks"]) for t in turns)
    act = np.array([realised(t) for t in turns])
    print(f"\n  out of fold, {len(turns)} turns ({n_events} picks)")
    print(f"  {'arm':22s} {'NLL/pick':>9s} {'P(assassin) pred/act':>22s} {'P(opp end) pred/act':>21s} "
          f"{'own words pred/act':>19s}")
    for arm in oof:
        p = np.array(oof[arm])
        print(f"  {arm:22s} {ll[arm] / n_events:9.4f} {p[:, 0].mean():10.4f} / {act[:, 0].mean():.4f} "
              f"{p[:, 1].mean():10.4f} / {act[:, 1].mean():.4f} {p[:, 2].mean():8.3f} / {act[:, 2].mean():.3f}")
    p = np.array(oof["uncalibrated"])
    q = np.array(oof["alpha + role offsets"])
    print("\n  assassin by predicted risk (uncalibrated bins): uncalibrated pred / calibrated pred / actual")
    for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 1.01)):
        m = (p[:, 0] >= lo) & (p[:, 0] < hi)
        if m.any():
            print(f"    [{lo:.2f}, {hi:.2f})  n={m.sum():4d}   {p[m, 0].mean():.4f} / {q[m, 0].mean():.4f} / "
                  f"{act[m, 0].mean():.4f} ({int(act[m, 0].sum())})")

    best = min(ARMS, key=lambda a: ll[a])
    final = fit(turns, ARMS[best])
    final.update({"arm": best, "booster": str(args.booster), "label": args.label, "turns": len(turns)})
    args.out.write_text(json.dumps(final, indent=1))
    print(f"\n  saved '{best}' fitted on all turns -> {args.out}: " +
          ", ".join(f"{k} {v:+.3f}" for k, v in final.items() if k.startswith(("alpha", "beta"))))


if __name__ == "__main__":
    main()
