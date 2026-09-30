"""Did the turns go the way the spymaster's listener said they would?

For every turn of a recorded suite matchup, rebuild the board as it stood,
ask the listener of the spymaster that gave the clue how it read that clue
(`listen`), and turn its scores into predictions for the number actually
announced (Plackett-Luce on frozen scores, codenames/pl_reward.py):
- P(the turn ends on the assassin);
- P(it ends on an opponent word);
- expected own words found.

These are then compared with what the guesser actually did, overall and by
bins of predicted risk.

**Why** (docs/versions/win_prob_listener.md). win_prob_listener beat the
incumbent while hitting the assassin more than twice as often (25 vs 11).
Two explanations fit:
- the win-probability objective prices the assassin lower than the old
  cost of 10 words, which is a deliberate trade;
- the listener underrates the assassin risk of the clues it chose, which is
  an error that search against the listener would seek out.

Calibration on the chosen clues tells them apart. It is also the
optimiser's-curse check: the argmax clue is where the listener's errors
are most likely to favour it.

    python scripts/tools/check_turn_calibration.py win_prob_listener:196b0da1ceec learned_listener:0e07db5a45d1 \\
        --param win_prob_listener turn_model=frozen
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_GPTOSS = "93f5a196b26054d1"


def build(name: str, params: dict):
    from codenames.spymasters.registry import spymaster_spec

    cls, kw = spymaster_spec(name)
    return cls(**{**kw, **params})


def predictions(spymaster, board, clue: str, k: int, sims) -> dict | None:
    from codenames.game import Role
    from codenames.pl_reward import gain_and_penalty

    r = spymaster.listen(board, clue, sims)
    if r is None:
        return None
    roles = r["roles"]
    s = np.asarray(r["scores"], dtype=np.float64)
    n_own = sum(x == Role.OWN for x in roles)
    k = min(k, n_own)
    bad_roles = roles[n_own:]
    s_own, s_bad = s[None, :n_own], s[None, n_own:]
    out = {}
    for role in (Role.ASSASSIN, Role.OPPONENT):
        ind = np.array([x == role for x in bad_roles], dtype=np.float64)
        gain, pen = gain_and_penalty(s_own, s_bad, ind, k)
        out[role.value] = float(pen[0, k - 1])
    out["own"] = float(gain[0, k - 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("challenger", help="spymaster id as recorded, name:hash")
    ap.add_argument("opponent", help="spymaster id as recorded, name:hash")
    ap.add_argument("--suite", default=SUITE_GPTOSS)
    ap.add_argument("--param", nargs=2, action="append", default=[], metavar=("NAME", "KEY=VALUE"),
                    help="a constructor parameter for one of the two spymasters")
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    args = ap.parse_args()

    from codenames.board import Board, Card, OpponentBoardView
    from codenames.game import Role
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    names = {sid: sid.split(":")[0] for sid in (args.challenger, args.opponent)}
    params: dict[str, dict] = {n: {} for n in names.values()}
    for name, kv in args.param:
        key, value = kv.split("=", 1)
        params[name][key] = value
    models = {sid: build(n, params[n]) for sid, n in names.items()}

    rows: dict[str, list] = {sid: [] for sid in names}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    for sm_id, board_json, turns_json in conn.execute(
            "SELECT spymaster_id, board, turns FROM game_records WHERE suite_id = ?", (args.suite,)):
        seats = dict(part.split("=", 1) for part in (sm_id or "").split(","))
        if set(seats.values()) != set(names):
            continue
        by_role = json.loads(board_json)
        cards = tuple(Card(w, Role(r)) for r, ws in by_role.items() for w in ws)
        board = Board(cards=cards, seed=0)
        for t in json.loads(turns_json):
            sid = seats[t["team"]]
            view = board if t["team"] == "A" else OpponentBoardView(board)
            p = predictions(models[sid], view, t["clue"], t["number"], sims)
            got = [role for _, role in t["guesses"]]
            if p is not None:
                rows[sid].append((p["assassin"], p["opponent"], p["own"],
                                  float("assassin" in got), float(got[-1:] == ["opponent"]),
                                  float(sum(r == "own" for r in got))))
            for w, _ in t["guesses"]:
                board.reveal(w)
    conn.close()

    for sid, rs in rows.items():
        a = np.array(rs)
        print(f"\n{sid}   {len(a)} turns")
        print(f"  {'':22s} {'predicted':>10s} {'actual':>8s}")
        for i, lab in ((0, "P(assassin)"), (1, "P(ends on opponent)"), (2, "own words per turn")):
            print(f"  {lab:22s} {a[:, i].mean():10.4f} {a[:, i + 3].mean():8.4f}")
        print("  assassin, by predicted risk:")
        for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 1.01)):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if m.any():
                print(f"    [{lo:.2f}, {hi:.2f})  n={m.sum():4d}   predicted {a[m, 0].mean():.4f}   "
                      f"actual {a[m, 3].mean():.4f} ({int(a[m, 3].sum())})")


if __name__ == "__main__":
    main()
