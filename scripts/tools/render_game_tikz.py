"""Render one recorded game as TikZ board diagrams for the paper.

Shows, for each of one spymaster's turns: the board as it stood, the clue and
number it gave, which own words it meant, and what the guesser actually did.

The intended words are not stored in the game record -- only the clue is -- so
they are recovered the same way the model chose them: for `expected_words`, the
clue's `number` is its own best k, and the words it meant are the k unrevealed
own words with the highest z-score for that clue. That is deterministic given
the board state, so replaying the turn log reproduces them exactly.

Usage:
    python scripts/tools/render_game_tikz.py --seed 43 --side A
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, Role
from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

FILL = {"own": "cnblue", "opponent": "cnred", "neutral": "cntan", "assassin": "cnblack"}
TEXTCOL = {"own": "white", "opponent": "white", "neutral": "black", "assassin": "white"}


def esc(w: str) -> str:
    return w.replace("&", "\\&").replace("_", "\\_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--side", default="A", choices=["A", "B"])
    ap.add_argument("--label-like", default="expected_words[sigma=1.5]-vs-centroid%")
    ap.add_argument("--space", default="numberbatch")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    row = con.execute(
        "select label, turns from game_records where seed = ? and label like ? order by id desc limit 1",
        (args.seed, args.label_like),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no game at seed {args.seed}")
    turns = json.loads(row[1])

    board = Board.generate(seed=args.seed)
    role_of = {c.word: c.role for c in board.cards}
    words = [c.word for c in board.cards]
    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    space_i = sims._space_index(args.space)
    clue_index = {w.lower(): i for i, w in enumerate(stats.clue_words)}

    def role_name(w: str) -> str:
        r = role_of[w]
        if r == Role.OWN:
            return "own" if args.side == "A" else "opponent"
        if r == Role.OPPONENT:
            return "opponent" if args.side == "A" else "own"
        return "assassin" if r == Role.ASSASSIN else "neutral"

    def intended(clue: str, k: int, revealed: set[str]) -> list[str]:
        """The k unrevealed own words the model scored highest for this clue."""
        ci = clue_index.get(clue.lower())
        mine = [w for w in words if role_name(w) == "own" and w not in revealed]
        if ci is None or not mine:
            return mine[:k]
        idx = [sims.board_index[w.lower()] for w in mine]
        z = (np.asarray(sims.tensor[ci, idx, space_i], dtype=np.float64)
             - float(stats.mean[ci, space_i])) / float(stats.std[ci, space_i])
        return [mine[i] for i in np.argsort(-z)[:k]]

    out: list[str] = []
    revealed: set[str] = set()
    n = 0
    for t in turns:
        side = str(t.get("team", "")).upper()
        if side != args.side:
            for w, _ in t.get("guesses", []):
                revealed.add(w)
            continue
        n += 1
        clue, k = t["clue"], int(t.get("number", 1))
        want = set(intended(clue, k, revealed))
        got = t.get("guesses", [])

        out.append("\\begin{figure}[H]\n\\centering\n\\begin{tikzpicture}[x=2.7cm, y=0.92cm]")
        for i, w in enumerate(words):
            r, c = divmod(i, 5)
            rn = role_name(w)
            done = w in revealed
            fill = f"{FILL[rn]}!22" if done else FILL[rn]
            txt = "black!35" if done else TEXTCOL[rn]
            style = (f"draw=black!25, line width=0.3pt, fill={fill}, "
                     f"minimum width=2.55cm, minimum height=0.8cm, inner sep=1pt, rounded corners=1.5pt")
            if w in want and not done:
                style = style.replace("draw=black!25, line width=0.3pt", "draw=cngold, line width=1.6pt")
            out.append(f"  \\node[{style}] at ({c},{-r}) "
                       f"{{\\textcolor{{{txt}}}{{\\scriptsize\\textsf{{{esc(w)}}}}}}};")
        out.append("\\end{tikzpicture}")

        picked = ", ".join(
            f"\\textbf{{{esc(w)}}}" if r == "own" else f"{esc(w)} \\textit{{({r})}}" for w, r in got
        ) or "nothing"
        out.append(
            "\\caption*{\\footnotesize \\textbf{Turn " + str(n) + "} --- clue \\textsc{" + esc(clue) + "} "
            + str(k) + ". Guesser took: " + picked + ".}"
        )
        out.append("\\end{figure}\n")

        for w, _ in got:
            revealed.add(w)

    print("\n".join(out))


if __name__ == "__main__":
    main()
