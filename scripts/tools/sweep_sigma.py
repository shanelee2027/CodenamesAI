"""Fit sigma against a real guesser.

`sigma` is the one free parameter of the `expected_words` objective: the
scale of the perturbation between our per-clue z and the guesser's
perceived ranking (docs/clue-selection-theory.tex, Setup). Every
value picked so far came from eyeballing the resulting clue numbers,
which is a proxy for the thing that matters and not the thing itself.

This measures it. For each candidate sigma the spymaster plays the same
positions, an LLM guesser ranks the board, and the turn is scored the way
codenames/game.py would -- own words until the first non-own word, then
`ROLE_REWARD` for whatever ended it. The sigma that maximizes realized
reward is the answer, because realized reward is what the objective
exists to maximize.

Two things worth reading off the sweep besides the peak:

- **Calibration.** The announced number is a claim: "I expect the guesser
  to reach this many." Comparing it against what the guesser delivers
  says whether the model is over- or under-confident, and in which
  direction sigma should move. A calibrated sigma has delivered ~=
  announced.
- **The extremes.** sigma -> 0 says the guesser sees exactly what we see,
  so claim the maximum every time; sigma -> infinity says the guesser is
  unpredictable, so claim one word and take the safest. Both should
  saturate. If they don't, something is wrong with the objective rather
  than with sigma.

Costs money: one call per distinct (clue, candidates, number) triple.
Adjacent sigmas often agree on the clue, and `number` is part of the
cache key, so the true call count is well below positions x sigmas --
`--dry-run` reports it exactly before anything is spent.

Positions come from a seed range disjoint from `configs/eval_suite.json`.
Fitting sigma against the frozen suite would spend it as an instrument:
the headline number would then be reported on boards the model was tuned
against.

Usage:
    python scripts/tools/sweep_sigma.py --dry-run -n 100
    python scripts/tools/sweep_sigma.py -n 100 --model claude-sonnet-5:medium
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, Role, load_holdout_wordlist
from codenames.env import load_env
from codenames.game import ROLE_REWARD
from codenames.guessers.llm import LLMGuesser
from codenames.llm_store import LLMResponseCache
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.expected_words import ExpectedWordsSpymaster

CACHE_PATH = PROJECT_ROOT / "cache" / "llm_store.db"

# Spans three orders of magnitude on purpose. The interesting region is
# 1.5-3.0, but the saturating ends are what show the objective behaving
# as the theory predicts rather than merely producing a number.
DEFAULT_SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]

# Disjoint from configs/eval_suite.json -- see the module docstring.
SEED0 = 5000


@dataclass
class Turn:
    board: Board
    clue: str
    number: int
    candidates: list[str]


def positions(n: int, seed0: int = SEED0) -> list[Board]:
    """Boards spanning the arc of a game, not just openings: sigma's
    effect on the announced number depends on how many own words are
    still live."""
    vocab = load_holdout_wordlist()
    out = []
    for i in range(n):
        board = Board.generate(seed0 + i, vocabulary=vocab)
        rng = random.Random(seed0 + i)
        for w in rng.sample(list(board.words), (i * 2) % 13):
            board.reveal(w)
        if board.remaining(Role.OWN) >= 2:
            out.append(board)
    return out


def turns_for(sigma: float, boards: list[Board], sims: SimilarityTensor) -> list[Turn]:
    sm = ExpectedWordsSpymaster(space="numberbatch", sigma=sigma, max_rarity=10.0)
    out = []
    for b in boards:
        ctx = TurnContext(board=b, turn_index=len(b.revealed))
        clue, number, _ = sm.top_clues(ctx, sims, 1)[0]
        out.append(Turn(board=b, clue=clue, number=number,
                        candidates=[w for w in b.words if not b.is_revealed(w)]))
    return out


def score(turn: Turn, ranking: list[str]) -> dict:
    """Exactly codenames/game.py::play_turn's accounting."""
    own, cause = 0, None
    for word in ranking[: turn.number]:
        role = turn.board.role_of(word)
        if role is not Role.OWN:
            cause = role
            break
        own += 1
    reward = own * ROLE_REWARD[Role.OWN] + (ROLE_REWARD[cause] if cause else 0.0)
    return {"own": own, "cause": cause.value if cause else "used the number",
            "reward": reward, "announced": turn.number}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--positions", type=int, default=100)
    ap.add_argument("--model", default="claude-sonnet-5:medium")
    ap.add_argument("--sigma", type=float, action="append", default=None)
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent API calls (default 12)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sigmas = sorted(args.sigma) if args.sigma else DEFAULT_SIGMAS
    model, _, effort = args.model.partition(":")
    effort = None if effort in ("", "none") else effort

    sims = SimilarityTensor.load()
    boards = positions(args.positions)
    by_sigma = {s: turns_for(s, boards, sims) for s in sigmas}

    triples = {(t.clue, tuple(t.candidates), t.number)
               for turns in by_sigma.values() for t in turns}
    cache = LLMResponseCache(CACHE_PATH)
    key = model if effort is None else f"{model}+effort={effort}"
    have = sum(1 for c, w, n in triples if cache.get(key, c, w, n) is not None)
    print(f"positions {len(boards)}   sigmas {sigmas}   model {args.model}")
    print(f"turns {len(boards) * len(sigmas)}   distinct prompts {len(triples)}   "
          f"cached {have}   to buy {len(triples) - have}")

    if args.dry_run:
        print("\nDry run: nothing sent.")
        return

    load_env()
    guesser = LLMGuesser(model=model, effort=effort, cache_path=CACHE_PATH)

    # Fetch each distinct prompt once, concurrently. Deduplicating first
    # matters as much as the concurrency: adjacent sigmas usually agree on
    # the clue and differ only in the announced number, so 1000 turns
    # collapse to ~500 prompts. LLMGuesser holds its lock only around the
    # shared dict and lazy client construction, never across the network
    # call, specifically so these can overlap (see its __init__).
    ordered = sorted(triples)

    def fetch(triple):
        clue, cands, number = triple
        return triple, guesser.rank_candidates(clue, list(cands), sims, number=number)

    rankings: dict[tuple, list[str]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for triple, ranking in pool.map(fetch, ordered):
            rankings[triple] = ranking
            done += 1
            if done % 50 == 0 or done == len(ordered):
                print(f"  fetched {done}/{len(ordered)}", flush=True)

    rows: dict[float, list[dict]] = {
        s: [score(t, rankings[(t.clue, tuple(t.candidates), t.number)])
            for t in by_sigma[s]]
        for s in sigmas
    }

    n = len(boards)
    print(f"\n{'sigma':>6}{'announced':>11}{'delivered':>11}{'gap':>7}"
          f"{'reward':>9}{'assassin':>10}{'opp':>6}{'neut':>6}{'clean':>7}")
    best = None
    for s in sigmas:
        r = rows[s]
        ann = sum(x["announced"] for x in r) / n
        del_ = sum(x["own"] for x in r) / n
        rew = sum(x["reward"] for x in r) / n
        c = {k: sum(1 for x in r if x["cause"] == k)
             for k in ("assassin", "opponent", "neutral", "used the number")}
        print(f"{s:>6}{ann:>11.2f}{del_:>11.2f}{del_ - ann:>+7.2f}{rew:>9.3f}"
              f"{c['assassin']:>10}{c['opponent']:>6}{c['neutral']:>6}"
              f"{c['used the number']:>7}")
        if best is None or rew > best[1]:
            best = (s, rew)

    print(f"\nbest realized reward: sigma = {best[0]} at {best[1]:+.3f} per turn")
    if args.out:
        args.out.write_text(json.dumps(
            {str(s): rows[s] for s in sigmas}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
