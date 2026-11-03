"""Comfort as agreement: do the words gpt-oss calls clear match the words the
clue was meant for? The comfort arena's stored games (no API calls).

**Comfort** (scripts/tools/plot_comfort_frontier.py) is min(|clear set|, k)
/ k: how much of the number the two-step guesser's clear set covers. It
counts the clear words, not which they are: a clue for 3 whose clear set is
three bystanders scores 1.

**Agreement** is |T ∩ G| / k for a clue for k:
- T, the words the clue was meant for: the k own words a fixed reference
  listener rates highest at the first pick (the play server's "Meant for").
  One reference for every model, online_vocab_listener's STOP listener, so
  that all models are judged against the same reading of their clues and
  a model's own listener cannot make its targets look easy.
- G, the guesser's first k: its clear set ("guess" relatedness group, asked
  without the number), in its stored ranking, cut at k. A clear set shorter
  than k leaves the rest empty, as a STOP would.
It is |T Δ G| = 2 (k − |T ∩ G|) rescaled, so the symmetric difference ranks
clues the same way. Also reported: own cover, |own ∩ G| / k, which counts
our words in G whether or not they were meant.

Per model: its turns on the suite, each position once (as the scoreboard),
means with a 95% bootstrap over boards.

    python scripts/tools/comfort_agreement.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

import plot_comfort_frontier as F  # noqa: E402
import replay_leftover_plus_one as R  # noqa: E402

REFERENCE = "online_vocab_listener"


def main() -> None:
    from codenames.board import Board, OpponentBoardView, Role, load_holdout_wordlist
    from codenames.guessers.base import STOP
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    sims = SimilarityTensor.load()
    cls, kw = spymaster_spec(REFERENCE)
    ref = cls(**kw)
    guesser = R.offline_guesser()
    vocab = load_holdout_wordlist()
    con = sqlite3.connect(f"file:{F.DB}?mode=ro", uri=True)
    rows = con.execute("SELECT spymaster_id, seed, turns, winner FROM game_records WHERE suite_id = ?",
                       (R.SUITE,)).fetchall()
    stats = defaultdict(lambda: {m: defaultdict(list) for m in ("comfort", "agree", "own", "k", "win")})
    seen: set = set()
    boards: dict = {}
    missing, t0 = 0, time.time()
    for sid, seed, tj, winner in rows:
        seat = dict(kv.split("=", 1) for kv in sid.split(","))
        pair = tuple(sorted(seat.values()))
        for team in "AB":
            stats[seat[team]]["win"][(pair, seed)].append(float(winner == team))
        if seed not in boards:
            boards[seed] = Board.generate(seed=seed, vocabulary=vocab)
        board = boards[seed]
        revealed: set = set()
        for t in json.loads(tj):
            pos = (sid, seed, t["team"], frozenset(revealed), t["clue"], int(t["number"]))
            if pos in seen or not t["clue"]:
                revealed |= {w for w, _ in t["guesses"]}
                continue
            seen.add(pos)
            k = int(t["number"])
            left = [w for w in board.words if w not in revealed]
            board.revealed = set(revealed)
            view = board if t["team"] == "A" else OpponentBoardView(board)
            try:
                ranked = guesser.rank_candidates(t["clue"], left, sims, number=k)
            except RuntimeError:
                missing += 1
                revealed |= {w for w, _ in t["guesses"]}
                continue
            clear = ranked[:ranked.index(STOP)] if STOP in ranked else ranked
            G = set(clear[:k])
            got = ref.listen(view, t["clue"], sims)
            if got is None:
                missing += 1
                revealed |= {w for w, _ in t["guesses"]}
                continue
            own = [(w, s) for w, r, s in zip(got["words"], got["roles"], got["scores"]) if r is Role.OWN]
            T = {w for w, _ in sorted(own, key=lambda x: -x[1])[:k]}
            s = stats[seat[t["team"]]]
            s["comfort"][seed].append(min(len(clear), k) / k)
            s["agree"][seed].append(len(T & G) / k)
            s["own"][seed].append(len(G & {w for w, _ in own}) / k)
            s["k"][seed].append(k)
            revealed |= {w for w, _ in t["guesses"]}

    print(f"{len(seen)} positions, {missing} skipped (no stored answer or no features), {time.time() - t0:.0f} s; "
          f"targets from {REFERENCE}")
    head = f"{'model':42s} {'win%':>6s} {'number':>6s} {'comfort':>16s} {'agreement':>22s} {'own cover':>16s}"
    print(head)
    out = []
    for m, s in stats.items():
        if not s["agree"]:
            continue
        win = F.boot(s["win"])
        c, a, o = F.boot(s["comfort"]), F.boot(s["agree"]), F.boot(s["own"])
        num = np.mean([x for v in s["k"].values() for x in v])
        out.append((F.NAMES.get(m, m), win, num, c, a, o))
    for name, win, num, c, a, o in sorted(out, key=lambda r: -r[4][0]):
        print(f"{name:42s} {win[0]:6.1%} {num:6.2f} {c[0]:6.3f} [{c[1]:.3f},{c[2]:.3f}] "
              f"{a[0]:8.3f} [{a[1]:.3f},{a[2]:.3f}] {o[0]:6.3f} [{o[1]:.3f},{o[2]:.3f}]")
    x = np.array([r[3][0] for r in out])
    y = np.array([r[4][0] for r in out])
    rank = lambda v: np.argsort(np.argsort(v))           # noqa: E731
    print(f"across {len(out)} models: Spearman(comfort, agreement) {np.corrcoef(rank(x), rank(y))[0, 1]:.3f}; "
          f"Spearman(win, agreement) "
          f"{np.corrcoef(rank(np.array([r[1][0] for r in out])), rank(y))[0, 1]:.3f}")


if __name__ == "__main__":
    main()
