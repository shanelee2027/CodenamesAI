"""Compare LLM guesser models on the same real turns.

The eval suite plays whole games, which is the right instrument for a
final number and the wrong one for "is the cheaper model good enough" --
game outcomes fold guesser quality together with luck about which boards
got hard, and cost tens of dollars to resolve.

This scores one turn at a time instead. Each model is handed the exact
prompt `LLMGuesser` sends for a real position, and its ranking is scored
the way `play_turn` would: walk the ranking, count own words until the
first non-own word, and record what that word was. That is the whole of
what a guesser contributes to a game, measured directly.

Reported per model:

- **own words revealed** -- the turn's actual yield, capped at the
  announced number, exactly as the game scores it.
- **cause** -- what ended the turn (neutral / opponent / assassin /
  exhausted the number).
- **intended hit rate** -- of the words the spymaster meant, how many the
  model actually took. Separates "the clue was bad" from "the model
  misread a good clue", which the yield alone cannot.
- **assassin rank** -- where the assassin sat in the ranking. A model can
  score identically to another this turn and be one place away from
  losing the game.

Rankings come through `LLMGuesser` with the shared disk cache, so a
repeat run of the same positions costs nothing and every response stays
available for later analysis.

Usage:
    python scripts/tools/compare_guesser_models.py --dry-run
    python scripts/tools/compare_guesser_models.py -n 25 \
        --model claude-opus-5:medium --model claude-sonnet-5:none
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import MAX_CLUE_NUMBER, Board, Role, load_holdout_wordlist
from codenames.clue_stats import ClueStats
from codenames.env import load_env
from codenames.guessers.llm import LLMGuesser
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.registry import spymaster_spec

CACHE_PATH = PROJECT_ROOT / "cache" / "llm_store.db"


def _z_for(clue: str, words: list[str], sims: SimilarityTensor) -> dict[str, float]:
    """Per-clue standardized similarity -- the quantity the spymaster
    scores on, so "what it intended" is read off the same scale it used."""
    stats = _clue_stats()
    ci = sims.clue_index[clue.lower()]
    si = stats.space_index(spymaster_spec("expected_words")[1].get("space", "numberbatch"))
    mu, sd = float(stats.mean[ci, si]), float(stats.std[ci, si])
    return {w: (float(sims.tensor[ci, sims.board_index[w.lower()], si]) - mu) / sd
            for w in words}


_STATS: ClueStats | None = None


def _clue_stats() -> ClueStats:
    global _STATS
    if _STATS is None:
        _STATS = ClueStats.load()
    return _STATS


@dataclass
class Position:
    board: Board
    clue: str
    number: int
    candidates: list[str]
    intended: list[str]


def collect_positions(n: int, sims: SimilarityTensor, seed0: int = 5000) -> list[Position]:
    """Real positions across the game, not just openings: a guesser's job
    changes once half the board is gone."""
    cls, kwargs = spymaster_spec("expected_words")
    spymaster = cls(**kwargs)
    vocab = load_holdout_wordlist()
    out = []
    for i in range(n):
        board = Board.generate(seed0 + i, vocabulary=vocab)
        rng = random.Random(seed0 + i)
        # 0, 2, 4, ... revealed, cycling, so the sample spans the arc of a
        # game rather than clustering at one stage.
        for w in rng.sample(list(board.words), (i * 2) % 13):
            board.reveal(w)
        if board.remaining(Role.OWN) < 2:
            continue
        ctx = TurnContext(board=board, turn_index=len(board.revealed))
        clue, number, _ = spymaster.top_clues(ctx, sims, 1)[0]
        candidates = [w for w in board.words if not board.is_revealed(w)]
        # What the spymaster was aiming at: the `number` own words the
        # clue scores highest, ranked by the same per-clue z the spymaster
        # scores on. Ranking by z is the whole point -- board order says
        # nothing about what the clue meant. Reporting only, never scoring.
        z = _z_for(clue, candidates, sims)
        own = sorted((w for w in candidates if board.role_of(w) is Role.OWN),
                     key=lambda w: -z[w])
        out.append(Position(board=board, clue=clue, number=number,
                            candidates=candidates, intended=own[:number]))
    return out


def score_turn(pos: Position, ranking: list[str]) -> dict:
    """Replay the turn exactly as codenames/game.py::play_turn would."""
    revealed_own, cause, guesses = 0, "used the number", []
    for word in ranking[: pos.number]:
        role = pos.board.role_of(word)
        guesses.append(word)
        if role is not Role.OWN:
            cause = role.value
            break
        revealed_own += 1
    assassin = next((w for w in pos.candidates
                     if pos.board.role_of(w) is Role.ASSASSIN), None)
    return {
        "own_revealed": revealed_own,
        "cause": cause,
        "guesses": guesses,
        "intended_hits": sum(1 for w in guesses if w in pos.intended),
        "assassin_rank": ranking.index(assassin) + 1 if assassin in ranking else None,
        "ranking": ranking,
    }


def parse_model(spec: str) -> tuple[str, str | None]:
    """`name` or `name:effort`; `none` means the thinking-disabled path."""
    model, _, effort = spec.partition(":")
    return model, (None if effort in ("", "none") else effort)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--positions", type=int, default=25)
    ap.add_argument("--model", action="append",
                    default=None, help="repeatable; `name` or `name:effort`")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    specs = args.model or ["claude-opus-5:medium", "claude-sonnet-5:none"]
    sims = SimilarityTensor.load()
    positions = collect_positions(args.positions, sims)

    print(f"positions {len(positions)}   models {specs}")
    print(f"{'seed':>6}  {'clue':<16}{'n':>2}  {'cand':>5}  intended")
    for p in positions:
        print(f"{p.board.seed:>6}  {p.clue:<16}{p.number:>2}  {len(p.candidates):>5}  "
              f"{', '.join(p.intended)}")
    if args.dry_run:
        print(f"\nDry run: no requests. Would send {len(positions) * len(specs)}.")
        return

    load_env()
    results: dict[str, list[dict]] = {}
    for spec in specs:
        model, effort = parse_model(spec)
        guesser = LLMGuesser(model=model, effort=effort, cache_path=CACHE_PATH)
        rows = []
        for p in positions:
            ranking = guesser.rank_candidates(p.clue, p.candidates, sims, number=p.number)
            row = score_turn(p, ranking)
            row.update(seed=p.board.seed, clue=p.clue, number=p.number)
            rows.append(row)
        results[spec] = rows

    n = len(positions)
    print(f"\n{'model':<26}{'own/turn':>9}{'intended':>10}{'assassin':>10}"
          f"{'opp':>6}{'neut':>6}{'clean':>7}")
    for spec, rows in results.items():
        own = sum(r["own_revealed"] for r in rows) / n
        possible = sum(min(p.number, MAX_CLUE_NUMBER) for p in positions)
        hits = sum(r["intended_hits"] for r in rows) / possible
        counts = {c: sum(1 for r in rows if r["cause"] == c)
                  for c in ("assassin", "opponent", "neutral", "used the number")}
        print(f"{spec:<26}{own:>9.2f}{hits:>9.0%}{counts['assassin']:>10}"
              f"{counts['opponent']:>6}{counts['neutral']:>6}"
              f"{counts['used the number']:>7}")

    if len(results) == 2:
        a, b = results.values()
        agree = sum(1 for x, y in zip(a, b) if x["guesses"] == y["guesses"])
        same_top = sum(1 for x, y in zip(a, b)
                       if x["ranking"][0] == y["ranking"][0])
        print(f"\nidentical guess sequences: {agree}/{n}   same top word: {same_top}/{n}")
        print(f"\n{'seed':>6}  {'clue':<14}{'n':>2}  "
              f"{list(results)[0][:18]:<18}  {list(results)[1][:18]:<18}")
        for p, x, y in zip(positions, a, b):
            mark = "" if x["guesses"] == y["guesses"] else "  <-- differs"
            print(f"{p.board.seed:>6}  {p.clue:<14}{p.number:>2}  "
                  f"{x['own_revealed']}/{p.number} {x['cause']:<14}  "
                  f"{y['own_revealed']}/{p.number} {y['cause']:<14}{mark}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
