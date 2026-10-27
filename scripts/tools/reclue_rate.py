"""How often a spymaster re-clues its own words: the words its earlier clues
were meant for and the guesser has not found yet. Costs nothing -- no LLM.

Recorded games keep each clue, its number and the guesses, not the words the
clue was meant for. Those are recomputed here as the play server's "Meant
for" does (Engine.explain): the `number` own words the spymaster's listener
rates highest at the position the clue was given in.

For every clue after a team's first:
- *pending*: the own words earlier clues of that team were meant for that are
  still unrevealed;
- *re-clued*: this clue's targets that are pending;
- *chance*: the re-clued count expected if the targets were `number` own
  words drawn at random from those left, number * pending / own left. Late
  in a game few words are left, so some overlap is unavoidable; the
  comparison with chance is what says whether the model seeks it out or
  avoids it;
- *repeat*: the clue word is one the same team already gave.

    python scripts/tools/reclue_rate.py --label-has margin_value_overpromise_listener:0288c5698292
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, OpponentBoardView, Role, load_holdout_wordlist  # noqa: E402

SUITE = "dc9083567bfcf453"          # the comfort arena: holdout_v1_gptoss_subset_guess

_SIMS = None
_SMS: dict = {}


def _spymaster(name: str):
    from codenames.spymasters.registry import spymaster_spec

    if name not in _SMS:
        base, _, _ = name.partition(":")
        cls, kw = spymaster_spec(base)
        _SMS[name] = cls(**kw)
    return _SMS[name]


def targets(sm, view, clue: str, number: int) -> list[str]:
    """The `number` own words the spymaster's listener rates highest for `clue`."""
    got = sm.listen(view, clue, _SIMS)
    if got is None:                     # a clue with no features
        return []
    own = [(w, s) for w, r, s in zip(got["words"], got["roles"], got["scores"]) if r is Role.OWN]
    return [w for w, _ in sorted(own, key=lambda x: -x[1])[:number]]


def replay(game: tuple) -> list[dict]:
    """One row per clue: team, spymaster, clue, number, targets, pending, own left."""
    global _SIMS
    from codenames.similarity import SimilarityTensor

    if _SIMS is None:
        _SIMS = SimilarityTensor.load()
    (game_id, seed, label, board_json, turns_json), only = game
    rec = json.loads(board_json)
    board = Board.generate(seed, vocabulary=load_holdout_wordlist())
    if set(board.words_by_role(Role.OWN)) != set(rec["own"]):
        raise ValueError(f"game {game_id}: seed {seed} does not rebuild the recorded board")
    seats = dict(s.split("=", 1) for s in label.split("|", 1)[1].split(","))
    sent: dict[str, list[tuple[str, list[str]]]] = {"A": [], "B": []}
    rows = []
    for t in json.loads(turns_json):
        team, clue, number = t["team"], t["clue"], int(t["number"])
        if only and only not in seats[team]:        # the other seat: only its guesses matter
            for word, _ in t["guesses"]:
                board.reveal(word)
            continue
        view = board if team == "A" else OpponentBoardView(board)
        own_left = set(view.words_by_role(Role.OWN, unrevealed_only=True))
        pending = {w for _, ws in sent[team] for w in ws} & own_left
        meant = targets(_spymaster(seats[team]), view, clue, number) if clue else []
        rows.append({"game": game_id, "team": team, "spymaster": seats[team], "clue": clue, "number": number,
                     "targets": meant, "pending": sorted(pending), "own_left": len(own_left),
                     "repeat": any(c.lower() == (clue or "").lower() for c, _ in sent[team]),
                     "first": not sent[team]})
        sent[team].append((clue, meant))
        for word, _ in t["guesses"]:
            board.reveal(word)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "cache" / "llm_store.db")
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--label-has", required=True, help="games whose label contains this (a spymaster:hash)")
    ap.add_argument("--seat-has", default=None,
                    help="only the turns of the seat whose spymaster contains this: one model loaded per worker")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="write the per-clue rows here (jsonl)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    games = conn.execute("SELECT id, seed, label, board, turns FROM game_records WHERE suite_id = ? AND label LIKE ?",
                         (args.suite, f"%{args.label_has}%")).fetchall()
    conn.close()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for g in ex.map(replay, [(g, args.seat_has) for g in games], chunksize=4) for r in g]
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    print(f"{len(games)} games, {len(rows)} clues")
    print(f"{'spymaster':48s} {'clues':>6s} {'w/ pending':>10s} {'re-clued':>9s} {'chance':>7s} "
          f"{'any re-clue':>11s} {'repeat':>7s}")
    for name in sorted({r["spymaster"] for r in rows}):
        later = [r for r in rows if r["spymaster"] == name and not r["first"] and r["clue"]]
        with_p = [r for r in later if r["pending"]]
        hit = sum(len(set(r["targets"]) & set(r["pending"])) for r in with_p)
        chance = sum(r["number"] * len(r["pending"]) / r["own_left"] for r in with_p)
        anyhit = sum(bool(set(r["targets"]) & set(r["pending"])) for r in with_p)
        rep = sum(r["repeat"] for r in later)
        print(f"{name:48s} {len(later):6d} {len(with_p):10d} {hit:9d} {chance:7.1f} "
              f"{anyhit / max(len(with_p), 1):10.1%} {rep:7d}")


if __name__ == "__main__":
    main()
