"""leftover_plus_one_listener against online_vocab_listener on positions
with a leftover: the comfort arena's recorded games, at turns whose team's
previous turn was cut short by a wrong guess with words still unfound
(no API calls).

Reported: how often the clue changes, and how often each model's clue
re-clues the leftover -- its targets (the `number` own words the listener
rates highest, as the play server's "Meant for") include the own word the
previous clue's next pick rates highest.

    python scripts/tools/compare_leftover_clues.py [--n 60]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE = "dc9083567bfcf453"


def positions(db: Path, n: int, seed: int = 0) -> list[tuple]:
    """(seed, revealed, team, history) at turns with a leftover by the rule."""
    from codenames.board import Board, OpponentBoardView, Role, load_holdout_wordlist
    from codenames.spymasters.base import PastTurn

    vocab = load_holdout_wordlist()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT seed, turns FROM game_records WHERE suite_id = ? ORDER BY id", (SUITE,)).fetchall()
    rng = np.random.default_rng(seed)
    out = []
    for g in rng.permutation(len(rows)):
        board_seed, turns = rows[g][0], json.loads(rows[g][1])
        board = Board.generate(board_seed, vocabulary=vocab)
        revealed: set = set()
        for i, t in enumerate(turns):
            team = t["team"]
            mine = [k for k in range(i) if turns[k]["team"] == team and turns[k]["clue"]]
            if mine and turns[mine[-1]]["guesses"]:
                prev = turns[mine[-1]]
                view = board if team == "A" else OpponentBoardView(board)
                found = sum(r == "own" for _, r in prev["guesses"])
                if prev["guesses"][-1][1] != "own" and int(prev["number"]) > found:
                    hist = tuple(PastTurn(turns[k]["team"] == team, turns[k]["clue"], int(turns[k]["number"]),
                                          tuple(w for w, _ in turns[k]["guesses"])) for k in range(i))
                    if view.words_by_role(Role.OWN, unrevealed_only=True):
                        out.append((board_seed, set(revealed), team, hist))
                        break                       # one position per game
            revealed |= {w for w, _ in t["guesses"]}
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    from codenames.board import Board, OpponentBoardView, Role, load_holdout_wordlist
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.base import TurnContext
    from codenames.spymasters.registry import spymaster_spec

    sims = SimilarityTensor.load()
    models = {}
    for name in ("online_vocab_listener", "leftover_plus_one_listener"):
        cls, kw = spymaster_spec(name)
        models[name] = cls(**kw)
    new = models["leftover_plus_one_listener"]
    vocab = load_holdout_wordlist()
    changed = reclue_old = reclue_new = n = 0
    q_sum, t0 = 0.0, time.time()
    for seed, revealed, team, hist in positions(PROJECT_ROOT / "cache" / "llm_store.db", args.n):
        board = Board.generate(seed, vocabulary=vocab)
        board.revealed = revealed
        view = board if team == "A" else OpponentBoardView(board)
        got = new.leftover(view, hist, sims)
        if got is None:
            continue
        L, q = got
        own, words, _ = new._board(view)
        lead = own[int(np.argmax(L[:len(own)]))]
        ctx = TurnContext(board=view, turn_index=len(revealed), history=hist)
        clues = {}
        for name, sm in models.items():
            clue, number, _ = sm.top_clues(ctx, sims, 1)[0]
            s = sm._listener_scores(clue, own, number, sims)
            targets = [own[i] for i in np.argsort(-s)[:number]] if s is not None else []
            clues[name] = (clue, number, lead in targets)
        (c0, k0, r0), (c1, k1, r1) = clues.values()
        n += 1
        q_sum += q
        changed += (c0, k0) != (c1, k1)
        reclue_old += r0
        reclue_new += r1
        print(f"  {hist[-2].clue if len(hist) > 1 else '':>12s} -> leftover {lead:12s} p={L[own.index(lead)]:.2f}  "
              f"old {c0} {k0}{' (re-clue)' if r0 else ''}   new {c1} {k1}{' (re-clue)' if r1 else ''}")
    print(f"{n} positions, {time.time() - t0:.0f} s: clue changed on {changed}; re-clues the leftover: "
          f"old {reclue_old}, new {reclue_new}; mean q {q_sum / max(n, 1):.2f}")


if __name__ == "__main__":
    main()
