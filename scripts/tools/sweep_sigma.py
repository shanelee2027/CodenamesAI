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
import os
import random
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, Card, OpponentBoardView, Role, load_holdout_wordlist
from codenames.env import load_env
from codenames.game import ROLE_REWARD
from codenames.guessers.llm import LLMGuesser
from codenames.guessers.registry import DEFAULT_POOL_CONFIG as DEFAULT_POOL
from codenames.llm_store import LLMResponseCache
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
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



# --------------------------------------------------------------------------
# CPU parallelism.
#
# Scoring one board against the clue vocabulary takes ~530ms, and a full
# --simulate sweep needs ~7000 of them (6000 to play the games, 1000 to score
# the resulting positions) -- an hour serially. torch defaults to 8 intra-op
# threads here and they are nearly worthless on this shape of work: measured
# 604ms on one thread against 526ms on eight, a 1.15x return for 8 cores. One
# process per core with torch pinned to a single thread each is ~14x instead.
# --------------------------------------------------------------------------

_W: dict = {}


def _pool_init(cache_dir: Path) -> None:
    # Set these BEFORE importing torch. torch.set_num_threads(1) alone runs
    # after OpenMP has already built its pool, and OpenMP busy-waits by
    # default -- 16 workers x 8 spinning threads on 16 cores burned 43
    # minutes of kernel time in a 7m55s run. "spawn" gives each worker a
    # fresh interpreter, so setting them here lands before torch loads.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    import torch

    torch.set_num_threads(1)
    _W["sims"] = SimilarityTensor.load(cache_dir)
    _W["spymasters"] = {}
    _W["guessers"] = {}


def _spymaster(sigma: float):
    if sigma not in _W["spymasters"]:
        _W["spymasters"][sigma] = ExpectedWordsSpymaster(space="numberbatch", sigma=sigma, max_rarity=10.0)
    return _W["spymasters"][sigma]


def _clue_task(task):
    """(sigma, board_seed, team, revealed) -> the clue that sigma plays there."""
    from codenames.guessers.registry import load_pool  # noqa: F401  (kept warm in workers)

    sigma, board_seed, team, revealed = task
    board = Board.generate(seed=board_seed)
    for w in revealed:
        board.reveal(w)
    view = board if team == "A" else OpponentBoardView(board)
    ctx = TurnContext(board=view, turn_index=len(revealed))
    clue, number, _ = _spymaster(sigma).top_clues(ctx, _W["sims"], 1)[0]
    cands = tuple(w for w in view.words if not view.is_revealed(w))
    return board_seed, team, revealed, clue, number, cands


def _play_task(task):
    """(sigma, board_seed, guesser_name) -> one board's two seatings, as
    (team, revealed-before-turn) snapshots of this sigma's own turns."""
    from codenames.game import play_two_team_game
    from codenames.guessers.registry import load_pool
    from codenames.spymasters.centroid import CentroidSpymaster

    sigma, board_seed, guesser_name = task
    if guesser_name not in _W["guessers"]:
        _W["guessers"][guesser_name] = load_pool(DEFAULT_POOL)[guesser_name].guesser
    guesser = _W["guessers"][guesser_name]
    sm, opponent = _spymaster(sigma), CentroidSpymaster(seed=0)

    out = []
    for swap in (False, True):
        board = Board.generate(seed=board_seed)
        teams = ((sm, guesser), (opponent, guesser)) if swap else ((opponent, guesser), (sm, guesser))
        result = play_two_team_game(board, teams[0], teams[1], _W["sims"])
        ours = "A" if swap else "B"
        revealed: list[str] = []
        for tt in result.turns:
            if tt.team == ours:
                out.append((ours, tuple(revealed)))
            revealed.extend(w for w, _ in tt.turn.guesses)
    return board_seed, out


def _pool(workers: int):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_pool_init,
        initargs=(DEFAULT_CACHE_DIR,),
    )


def positions(n: int, seed0: int = SEED0) -> list[Board]:
    """**Superseded -- see `positions_from_games` and use --positions-from.**

    Reveals `(i * 2) % 13` cards chosen uniformly at random. That is a
    bad model of a game's arc, and it biased the sigma this sweep was
    built to choose. 16 of the 25 cards are distractors, so uniform
    reveals clear distractors ~1.8x faster than own words and the board
    gets *easier* as it progresses; real play takes an own word ~93% of
    the time, so own words go first and the distractor field stays dense.
    At a matched own-word count these boards carry 2-3 fewer live
    distractors than real ones, which inflates the announced number --
    measured 1.72 here against 1.16 in real games (see docs/log.md).

    Kept so the original sweep in docs/clue-selection-theory.tex can be
    reproduced, not because it should be used again."""
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


def positions_from_games(n: int, db_path: Path, label_like: str = "%", seed: int = 0) -> list:
    """Real positions: the board state before each turn of games already
    played and recorded by codenames/llm_store.py::GameRecordStore.

    This costs nothing -- the games are on disk -- and it reproduces the
    distribution real play actually visits, which `positions` above does
    not. A turn taken by team B is handed back as an OpponentBoardView,
    so the spymaster sees that team's own words as Role.OWN exactly as it
    did when the game was played.

    Sampled uniformly across all recorded turns rather than taking the
    first n, so the result spans openings through endgames in the
    proportion games actually produce them.

    One caveat worth stating: these positions were produced by games the
    models under test played, so they are not independent of those
    models. They are far closer to real play than random reveals, but a
    fully independent set would need games from a spymaster that is not
    being swept."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT seed, board, turns FROM game_records WHERE label LIKE ? ORDER BY id", (label_like,)
    ).fetchall()
    conn.close()

    snapshots = []
    for row in rows:
        by_role = json.loads(row["board"])
        # Regenerate from the seed rather than rebuilding from the stored
        # role groups. The record keeps only {role: [words]}, so rebuilding
        # from it orders the board own-first -- but `board.words` order is
        # what becomes the candidate list, which is both the LLM prompt's
        # word order and part of the response cache key. Grouped order
        # would present a board no game ever showed and miss every cached
        # response. Falls back to the grouped rebuild if the seed doesn't
        # reproduce the recorded roles (a different board vocabulary).
        regenerated = Board.generate(seed=row["seed"])
        if all(regenerated.role_of(w).value == role for role, ws in by_role.items() for w in ws):
            cards = regenerated.cards
        else:
            cards = tuple(Card(word=w, role=Role(role)) for role, words in by_role.items() for w in words)
        revealed: list[str] = []
        for turn in json.loads(row["turns"]):
            # Rebuild from scratch per turn: Board.reveal mutates, and each
            # snapshot has to stay independent of the ones after it.
            board = Board(cards=cards, seed=0)
            for w in revealed:
                board.reveal(w)
            if board.remaining(Role.OWN if turn["team"] == "A" else Role.OPPONENT) >= 1:
                snapshots.append(board if turn["team"] == "A" else OpponentBoardView(board))
            revealed.extend(w for w, _ in turn["guesses"])

    rng = random.Random(seed)
    return snapshots if len(snapshots) <= n else rng.sample(snapshots, n)


def positions_from_play(sigma: float, n_boards: int, guesser_name: str, sims: SimilarityTensor,
                        n: int | None = None, seed: int = 0) -> list:
    """Positions this sigma's OWN play reaches: plays `n_boards` boards
    (both seatings) of expected_words at this sigma against a fixed
    `centroid` opponent, with a free synthetic listener, and snapshots the
    board before each of its turns.

    This is what `positions_from_games` can't do. A cautious sigma that
    announces 1s and an aggressive one that announces 4s reach genuinely
    different boards, and scoring every sigma on one fixed position set
    erases that difference -- which is part of what choosing a sigma
    should be weighing. Here each sigma is scored under the distribution
    its own play produces.

    Costs nothing: only the LLM *scoring* of the resulting positions is
    paid for, and reaching a position needs no LLM at all. The same board
    seeds are used for every sigma, so the comparison stays paired at the
    board level even though the positions within a board differ.

    `guesser_name` should be a listener from a DIFFERENT embedding space
    than the spymaster scores in. Measured against the 100 recorded LLM
    games, noisy_wikipedia2vec reproduces their position distribution
    best (L1 0.074, own-word rate 91.7% vs the LLM's 93.2%);
    noisy_numberbatch is the same space expected_words scores in, so it
    is a near-perfect listener at 99.9% and its games run unrealistically
    clean."""
    from codenames.game import play_two_team_game
    from codenames.guessers.registry import load_pool
    from codenames.spymasters.centroid import CentroidSpymaster

    guesser = load_pool(DEFAULT_POOL)[guesser_name].guesser
    sm = ExpectedWordsSpymaster(space="numberbatch", sigma=sigma, max_rarity=10.0)
    opponent = CentroidSpymaster(seed=0)

    snapshots = []
    for board_seed in range(n_boards):
        for swap in (False, True):
            board = Board.generate(seed=board_seed)
            teams = ((sm, guesser), (opponent, guesser)) if swap else ((opponent, guesser), (sm, guesser))
            result = play_two_team_game(board, teams[0], teams[1], sims)
            ours = "A" if swap else "B"
            # Replay onto a fresh board: play_two_team_game mutated the
            # one above, and each snapshot must be independent.
            replay = Board.generate(seed=board_seed)
            revealed: list[str] = []
            for tt in result.turns:
                if tt.team == ours:
                    b = Board.generate(seed=board_seed)
                    for w in revealed:
                        b.reveal(w)
                    view = b if ours == "A" else OpponentBoardView(b)
                    if view.remaining(Role.OWN) >= 1:
                        snapshots.append(view)
                revealed.extend(w for w, _ in tt.turn.guesses)

    if n is None or len(snapshots) <= n:
        return snapshots
    return random.Random(seed).sample(snapshots, n)


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
    ap.add_argument(
        "--positions-from",
        type=Path,
        default=None,
        help="snapshot real positions from games recorded in this db (cache/llm_store.db) instead of "
        "revealing cards at random -- see positions_from_games. Strongly preferred: the random-reveal "
        "default builds boards easier than real play and biases the chosen sigma (docs/log.md).",
    )
    ap.add_argument("--positions-label", default="%", help="restrict --positions-from to game labels matching this LIKE pattern")
    ap.add_argument(
        "--simulate",
        type=int,
        default=None,
        metavar="N_BOARDS",
        help="BEST OPTION. Generate each sigma's positions by simulating N_BOARDS boards of its own "
        "play (both seatings) against a fixed centroid opponent, with a free synthetic listener. "
        "Costs nothing extra -- only scoring the resulting positions is paid for. Each sigma is then "
        "measured under the board distribution its own play produces, which neither --positions-from "
        "nor the random-reveal default can do.",
    )
    ap.add_argument("--simulate-guesser", default="noisy_wikipedia2vec",
                    help="listener for --simulate. Use a DIFFERENT space than the spymaster scores in "
                         "(default matches the recorded LLM games' position distribution best).")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="processes for clue scoring and game simulation (default: os.cpu_count()). "
                         "Each pins torch to 1 thread -- torch's own 8 threads return only 1.15x here.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sigmas = sorted(args.sigma) if args.sigma else DEFAULT_SIGMAS
    model, _, effort = args.model.partition(":")
    effort = None if effort in ("", "none") else effort

    sims = SimilarityTensor.load()
    workers = args.cpu_workers or (os.cpu_count() or 8)

    if args.simulate is not None:
        # Fan out every (sigma, board) pair at once rather than a sigma at a
        # time: 10 sigmas x 25 boards is 250 independent games, and doing
        # them one sigma at a time leaves most cores idle at each stage.
        play_tasks = [(sg, seed, args.simulate_guesser) for sg in sigmas for seed in range(args.simulate)]
        snaps: dict[float, list] = {sg: [] for sg in sigmas}
        done = 0
        by_sigma = {sg: [] for sg in sigmas}
        with _pool(workers) as ex:
            for (sg, _, _), (board_seed, turns) in zip(play_tasks, ex.map(_play_task, play_tasks)):
                snaps[sg].extend((board_seed, team, rev) for team, rev in turns)
                done += 1
                if done % 50 == 0:
                    print(f"  simulated {done}/{len(play_tasks)} boards", flush=True)

            rng = random.Random(0)
            clue_tasks = []
            for sg in sigmas:
                picked = snaps[sg] if len(snaps[sg]) <= args.positions else rng.sample(snaps[sg], args.positions)
                clue_tasks.extend((sg, seed, team, rev) for seed, team, rev in picked)
                print(f"  sigma {sg}: {len(snaps[sg])} positions -> {len(picked)} sampled", flush=True)

            for (sg, *_), (board_seed, team, revealed, clue, number, cands) in zip(
                clue_tasks, ex.map(_clue_task, clue_tasks)
            ):
                board = Board.generate(seed=board_seed)
                for w in revealed:
                    board.reveal(w)
                view = board if team == "A" else OpponentBoardView(board)
                by_sigma[sg].append(Turn(board=view, clue=clue, number=number, candidates=list(cands)))
        boards = None
    else:
        if args.positions_from is not None:
            boards = positions_from_games(args.positions, args.positions_from, args.positions_label)
        else:
            boards = positions(args.positions)
        by_sigma = {s: turns_for(s, boards, sims) for s in sigmas}

    triples = {(t.clue, tuple(t.candidates), t.number)
               for turns in by_sigma.values() for t in turns}
    cache = LLMResponseCache(CACHE_PATH)
    key = model if effort is None else f"{model}+effort={effort}"
    have = sum(1 for c, w, n in triples if cache.get(key, c, w, n) is not None)
    n_turns = sum(len(t) for t in by_sigma.values())
    src = f"simulated per sigma ({args.simulate} boards)" if args.simulate else f"{len(boards)} shared"
    print(f"positions {src}   sigmas {sigmas}   model {args.model}")
    print(f"turns {n_turns}   distinct prompts {len(triples)}   "
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

    # Per sigma, not len(boards): with --simulate each sigma has its own
    # position set and there is no shared `boards` list at all.
    print(f"\n{'sigma':>6}{'n':>5}{'announced':>11}{'delivered':>11}{'gap':>7}"
          f"{'reward':>9}{'s.e.':>7}{'assassin':>10}{'opp':>6}{'neut':>6}{'clean':>7}")
    best = None
    for s in sigmas:
        r = rows[s]
        n = len(r)
        ann = sum(x["announced"] for x in r) / n
        del_ = sum(x["own"] for x in r) / n
        rew = sum(x["reward"] for x in r) / n
        var = sum((x["reward"] - rew) ** 2 for x in r) / (n - 1) if n > 1 else 0.0
        se = (var / n) ** 0.5
        c = {k: sum(1 for x in r if x["cause"] == k)
             for k in ("assassin", "opponent", "neutral", "used the number")}
        print(f"{s:>6}{n:>5}{ann:>11.2f}{del_:>11.2f}{del_ - ann:>+7.2f}{rew:>9.3f}{se:>7.3f}"
              f"{c['assassin']:>10}{c['opponent']:>6}{c['neutral']:>6}"
              f"{c['used the number']:>7}")
        if best is None or rew > best[1]:
            best = (s, rew, se)

    print(f"\nbest realized reward: sigma = {best[0]} at {best[1]:+.3f} +/- {best[2]:.3f} per turn")
    print("Compare sigmas against these standard errors -- neighbouring values")
    print("are often within noise of each other.")
    if args.out:
        args.out.write_text(json.dumps(
            {str(s): rows[s] for s in sigmas}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
