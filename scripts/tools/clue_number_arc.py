"""How the announced clue number falls as a team's own words run out, per sigma.

Costs nothing -- no LLM anywhere. Results are cached so the plotting notebook
(scratch/clue_number_arc.ipynb) never has to re-simulate.

**Why simulate rather than sample board states.** The words left late in a real
game are not a random subset of the nine: the obvious pairings went first, and
what remains is what nothing connected. A board holding five randomly chosen
own words is easier than one holding the five a real game failed to clue away.
`--twins` measures that gap against a matched control -- the same board with
the same number of cards revealed per role, but randomly chosen ones.

**The one rule change.** A miss ends the turn exactly as in Codenames, but
never ends the *game*: neutral, opponent and assassin reveals are absorbed and
play continues, so a single game traces the whole arc from 9 own words down to
1 instead of stopping at the first assassin.

Usage:
    python scripts/tools/clue_number_arc.py                    # full run, ~20 min on 16 cores
    python scripts/tools/clue_number_arc.py --n-games 10       # quick check
    python scripts/tools/clue_number_arc.py --reuse            # re-aggregate cached rows, no simulation
"""
from __future__ import annotations

import os

# Set before numpy/torch load, in this process and in every spawned worker.
#
# Thread pinning: torch's 8 intra-op threads return only ~1.15x on this shape
# of work (604ms vs 526ms per clue search) while occupying 8 cores, so 16
# single-threaded processes beat 2 eight-threaded ones by ~8x.
#
# malloc: gain_and_penalty allocates ~68MB transients ((n_cand, GRID_CELLS,
# n_words) float32) several times per clue search. glibc serves blocks that
# size with mmap and munmaps them on free, so many workers churning them
# produce a page-fault storm -- measured sys time at 0.93x user. Keeping them
# on the heap for reuse drops that to 0.04x.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "1073741824")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "1073741824")

import argparse
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.board import MAX_CLUE_NUMBER, Board, Role
from codenames.guessers.registry import load_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.expected_words import ExpectedWordsSpymaster

DEFAULT_SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
DEFAULT_TWIN_SIGMAS = [1.0, 2.5, 4.0]
DEFAULT_GUESSER = "noisy_numberbatch"
POOL_CONFIG = PROJECT_ROOT / "configs" / "guesser_pool.json"

# Raw per-turn rows are regenerable, so they live in the gitignored cache.
# The aggregate is small, citable, and worth keeping in git -- same split the
# sigma sweep uses (docs/sigma_sweep_simulated.json).
ROWS_OUT = PROJECT_ROOT / "cache" / "clue_number_arc_rows.json"
SUMMARY_OUT = PROJECT_ROOT / "docs" / "clue_number_arc.json"

_W: dict = {}


def _init(cache_dir: Path) -> None:
    import torch

    torch.set_num_threads(1)
    _W["sims"] = SimilarityTensor.load(cache_dir)
    _W["sm"], _W["g"] = {}, {}


def _sm(sigma: float) -> ExpectedWordsSpymaster:
    if sigma not in _W["sm"]:
        _W["sm"][sigma] = ExpectedWordsSpymaster(space="numberbatch", sigma=sigma, max_rarity=10.0)
    return _W["sm"][sigma]


def _guesser(name: str):
    if name not in _W["g"]:
        _W["g"][name] = load_pool(POOL_CONFIG)[name].guesser
    return _W["g"][name]


def play_arc(task):
    """(sigma, seed, guesser_name) -> one row per turn, until own words run out."""
    sigma, seed, guesser_name = task
    sims, sm, guesser = _W["sims"], _sm(sigma), _guesser(guesser_name)
    board = Board.generate(seed=seed)
    rows = []
    for _ in range(40):  # a stuck game would otherwise spin forever
        own_left = board.remaining(Role.OWN)
        if own_left == 0:
            break
        revealed_by_role = {
            r.value: sum(1 for c in board.cards if c.role is r and c.word in board.revealed)
            for r in Role
        }
        ctx = TurnContext(board=board, turn_index=len(board.revealed))
        clue, number = sm.give_clue(ctx, sims)
        candidates = [w for w in board.words if not board.is_revealed(w)]
        ranked = guesser.rank_candidates(clue, candidates, sims, number=number)
        got, stopped = 0, None
        for word in ranked[:number]:
            role = board.reveal(word)
            if role is not Role.OWN:
                stopped = role.value
                break
            got += 1
        rows.append({"sigma": sigma, "seed": seed, "own_before": own_left, "k": number,
                     "got": got, "stopped": stopped, "clue": clue,
                     "revealed_by_role": revealed_by_role})
    return rows


def shuffled_twin(task):
    """Matched control: same board, same counts revealed per role, random cards."""
    sigma, seed, revealed_by_role, twin_seed = task
    board = Board.generate(seed=seed)
    rng = random.Random(twin_seed)
    for role_value, count in revealed_by_role.items():
        pool = [c.word for c in board.cards if c.role.value == role_value]
        for w in rng.sample(pool, min(count, len(pool))):
            board.reveal(w)
    if board.remaining(Role.OWN) == 0:
        return None
    ctx = TurnContext(board=board, turn_index=len(board.revealed))
    _, number = _sm(sigma).give_clue(ctx, _W["sims"])
    return {"sigma": sigma, "seed": seed, "own_before": board.remaining(Role.OWN), "k": number}


def _mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, (var / n) ** 0.5, n


def aggregate(rows, twin_rows=None):
    """Per-(sigma, n) means, plus the per-sigma collapse.

    Two per-sigma numbers, because the pooled one is confounded: turns per game
    runs from ~3.4 at sigma=0.25 to ~13.5 at sigma=10 (a big announced number
    ends games fast), so pooling mixes "what this sigma announces" with "which
    board states it spends its turns in". The opening clue is the clean
    comparison -- exactly one per game, same count for every sigma.
    """
    arc, by = {}, {}
    for r in rows:
        by.setdefault((r["sigma"], r["own_before"]), []).append(r["k"])
    for (sg, n), ks in by.items():
        m, se, cnt = _mean_se(ks)
        arc.setdefault(str(sg), {})[str(n)] = {"mean": m, "se": se, "count": cnt}

    per_sigma = {}
    for sg in sorted({r["sigma"] for r in rows}):
        ks = [r["k"] for r in rows if r["sigma"] == sg]
        opening = [r["k"] for r in rows if r["sigma"] == sg and r["own_before"] == 9]
        n_games = len({r["seed"] for r in rows if r["sigma"] == sg})
        pm, pse, _ = _mean_se(ks)
        om, ose, _ = _mean_se(opening) if opening else (float("nan"), 0.0, 0)
        per_sigma[str(sg)] = {"pooled_mean": pm, "pooled_se": pse,
                              "opening_mean": om, "opening_se": ose,
                              "turns": len(ks), "games": n_games,
                              "turns_per_game": len(ks) / n_games}

    out = {"arc": arc, "per_sigma": per_sigma}
    if twin_rows:
        tw = {}
        for r in twin_rows:
            tw.setdefault((r["sigma"], r["own_before"]), []).append(r["k"])
        out["twin"] = {}
        for (sg, n), ks in tw.items():
            m, se, cnt = _mean_se(ks)
            out["twin"].setdefault(str(sg), {})[str(n)] = {"mean": m, "se": se, "count": cnt}
    return out


def _pool(workers):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init, initargs=(DEFAULT_CACHE_DIR,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, action="append", default=None, help="repeatable; default: the standard ten")
    ap.add_argument("--n-games", type=int, default=100, help="games per sigma")
    ap.add_argument("--guesser", default=DEFAULT_GUESSER,
                    help="listener. The default shares the spymaster's own space, so it is a near-perfect "
                         "one (~99.9%% own-word rate) -- deliberate, it keeps guesser error out of the arc.")
    ap.add_argument("--twin-sigma", type=float, action="append", default=None)
    ap.add_argument("--no-twins", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="default: os.cpu_count()")
    ap.add_argument("--rows-out", type=Path, default=ROWS_OUT)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    ap.add_argument("--reuse", action="store_true", help="re-aggregate cached rows instead of simulating")
    args = ap.parse_args()

    sigmas = sorted(args.sigma) if args.sigma else DEFAULT_SIGMAS
    twin_sigmas = [] if args.no_twins else sorted(args.twin_sigma) if args.twin_sigma else DEFAULT_TWIN_SIGMAS
    twin_sigmas = [s for s in twin_sigmas if s in sigmas]
    workers = args.workers or os.cpu_count() or 8

    if args.reuse:
        cached = json.loads(args.rows_out.read_text())
        rows, twin_rows = cached["rows"], cached.get("twin_rows", [])
        print(f"reusing {len(rows)} rows from {args.rows_out}")
    else:
        print(f"{len(sigmas)} sigmas x {args.n_games} games, guesser {args.guesser}, {workers} workers")
        tasks = [(sg, seed, args.guesser) for sg in sigmas for seed in range(args.n_games)]
        t0 = time.time()
        with _pool(workers) as ex:
            rows = [r for batch in ex.map(play_arc, tasks, chunksize=1) for r in batch]
            print(f"  arcs: {len(tasks)} games -> {len(rows)} turns in {time.time() - t0:.0f}s")

            twin_rows = []
            if twin_sigmas:
                tt = [(r["sigma"], r["seed"], r["revealed_by_role"], i)
                      for i, r in enumerate(rows) if r["sigma"] in twin_sigmas]
                t1 = time.time()
                # chunked: each twin is a single ~0.6s search, so per-task IPC shows up
                twin_rows = [t for t in ex.map(shuffled_twin, tt, chunksize=16) if t is not None]
                print(f"  twins: {len(tt)} tasks -> {len(twin_rows)} rows in {time.time() - t1:.0f}s")

        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        args.rows_out.write_text(json.dumps({"rows": rows, "twin_rows": twin_rows}))
        print(f"  wrote {args.rows_out} ({args.rows_out.stat().st_size / 1e6:.1f} MB)")

    summary = aggregate(rows, twin_rows)
    summary["config"] = {"sigmas": sigmas, "n_games": args.n_games, "guesser": args.guesser,
                         "twin_sigmas": twin_sigmas, "max_clue_number": MAX_CLUE_NUMBER}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=1))

    print(f"\n{'sigma':>6} {'pooled k':>9} {'opening k':>10} {'turns':>7} {'turns/game':>11}")
    print("-" * 48)
    for sg in sigmas:
        d = summary["per_sigma"].get(str(sg))
        if d is None:
            continue
        print(f"{sg:>6} {d['pooled_mean']:>9.2f} {d['opening_mean']:>10.2f} "
              f"{d['turns']:>7} {d['turns_per_game']:>11.1f}")
    print("\npooled weights every turn equally (what a game produces); opening is the")
    print("n=9 clue alone -- one per game, so the one comparison with no composition")
    print(f"difference across sigmas. Announced k is capped at min(n, {MAX_CLUE_NUMBER}).")
    print(f"\nwrote {args.summary_out}")


if __name__ == "__main__":
    main()
