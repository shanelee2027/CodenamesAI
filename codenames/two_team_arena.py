"""Two-team arenas: bulk-run codenames.game.play_two_team_game across many
seeded boards, in parallel worker processes.

**Head-to-head** (`run_two_team_matchup`) is the comparison every sweep and
the frozen eval suite make: two spymasters, one per side, sharing one
guesser, every board played twice with the sides swapped.

**Self-play** (`run_two_team_self_play`) puts the same spymaster+guesser
pair on both sides. Which team wins then says little -- mostly the 9-vs-8
first-move edge -- so it reports how often play ends on the assassin versus
a clean finish, with both teams' turns pooled. `guesser_name` can be
`MIXED_GUESSER` ("mixed"): each game draws one guesser uniformly from the
pool config.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import traceback
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from codenames.board import Board, Role
from codenames.game import DEFAULT_MAX_TURNS, TwoTeamGameResult, play_two_team_game
from codenames.guessers.registry import DEFAULT_POOL_CONFIG, build_guesser, training_pool
from codenames.llm_store import GameRecordStore, board_by_role
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

# Passed as `guesser_name` to mean "don't fix one guesser -- each game
# independently draws one, uniformly, from every guesser in the pool config".
MIXED_GUESSER = "mixed"


@dataclass
class TwoTeamSelfPlayResult:
    n_games: int
    assassin_rate: float  # fraction of games ending via either side hitting the assassin
    mean_half_turns_all: float  # both teams' turns combined, all games
    mean_half_turns_clean_finish: float | None  # same, games that ended by someone clearing their words
    guess_own_rate: float
    guess_opponent_rate: float
    guess_neutral_rate: float
    guess_assassin_rate: float
    mean_correct_per_clue: float  # own-word guesses per turn (the "k" a clue actually earned), pooled across both teams
    mean_clue_number: float  # announced clue number, pooled across both teams


def _new_stats() -> dict[str, float]:
    return {
        "games": 0,
        "assassin_endings": 0,
        "half_turns": 0,
        "half_turns_clean": 0,
        "clean_games": 0,
        "guesses": 0,
        "guess_own": 0,
        "guess_opponent": 0,
        "guess_neutral": 0,
        "guess_assassin": 0,
        "clues": 0,
        "clue_number_sum": 0,
        "correct_per_clue_sum": 0,
    }


def update_stats(s: dict[str, float], result: TwoTeamGameResult) -> None:
    s["games"] += 1
    s["half_turns"] += len(result.turns)
    if result.outcome == "loss":
        s["assassin_endings"] += 1
    else:
        s["clean_games"] += 1
        s["half_turns_clean"] += len(result.turns)
    for tt in result.turns:
        s["clues"] += 1
        s["clue_number_sum"] += tt.turn.number
        s["correct_per_clue_sum"] += sum(1 for _, role in tt.turn.guesses if role == Role.OWN)
        for _, role in tt.turn.guesses:
            s["guesses"] += 1
            s[f"guess_{role.value}"] += 1


def finalize_result(s: dict[str, float]) -> TwoTeamSelfPlayResult:
    return TwoTeamSelfPlayResult(
        n_games=int(s["games"]),
        assassin_rate=s["assassin_endings"] / s["games"],
        mean_half_turns_all=s["half_turns"] / s["games"],
        mean_half_turns_clean_finish=(s["half_turns_clean"] / s["clean_games"]) if s["clean_games"] else None,
        guess_own_rate=(s["guess_own"] / s["guesses"]) if s["guesses"] else 0.0,
        guess_opponent_rate=(s["guess_opponent"] / s["guesses"]) if s["guesses"] else 0.0,
        guess_neutral_rate=(s["guess_neutral"] / s["guesses"]) if s["guesses"] else 0.0,
        guess_assassin_rate=(s["guess_assassin"] / s["guesses"]) if s["guesses"] else 0.0,
        mean_correct_per_clue=(s["correct_per_clue_sum"] / s["clues"]) if s["clues"] else 0.0,
        mean_clue_number=(s["clue_number_sum"] / s["clues"]) if s["clues"] else 0.0,
    )


_WORKER_STATE: dict = {}

# Guards the one shared GameRecordStore when workers are threads. Under
# processes each worker had its own connection and needed no lock.
_RECORD_LOCK = threading.Lock()


def _worker_init(
    sims_cache_dir: Path,
    spymaster_cls: type,
    spymaster_kwargs: dict,
    guesser_pool_config: Path,
    guesser_name: str,
    max_turns: int,
    game_record_db: Path | None,
    run_label: str,
) -> None:
    # Constructs the spymaster fresh inside the worker rather than pickling
    # an existing instance across the process boundary.
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["spymaster"] = spymaster_cls(**spymaster_kwargs)
    _WORKER_STATE["guesser_name"] = guesser_name
    if guesser_name == MIXED_GUESSER:
        _WORKER_STATE["guesser_pool"] = list(training_pool(guesser_pool_config).values())
    else:
        _WORKER_STATE["guesser"] = build_guesser(guesser_name, guesser_pool_config)
    _WORKER_STATE["max_turns"] = max_turns
    # One WAL-mode SQLite connection per worker process (see
    # codenames/llm_store.py) -- concurrent writers to the same file are
    # safe, so this doesn't need any cross-process coordination.
    _WORKER_STATE["record_store"] = GameRecordStore(game_record_db) if game_record_db is not None else None
    _WORKER_STATE["run_label"] = run_label


def _play_task(seed: int) -> TwoTeamGameResult:
    state = _WORKER_STATE
    board = Board.generate(seed=seed)
    if state["guesser_name"] == MIXED_GUESSER:
        guesser = random.Random(seed).choice(state["guesser_pool"])
    else:
        guesser = state["guesser"]
    team = (state["spymaster"], guesser)
    # Snapshotted before any word is revealed -- play_two_team_game
    # mutates this same Board object in place.
    by_role = board_by_role(board) if state["record_store"] is not None else None
    result = play_two_team_game(board, team, team, state["sims"], max_turns=state["max_turns"])
    if state["record_store"] is not None:
        with _RECORD_LOCK:
            state["record_store"].add_game(by_role, result, label=state["run_label"])
    return result


def run_two_team_self_play(
    spymaster_cls: type,
    spymaster_kwargs: dict,
    guesser_pool_config: Path,
    guesser_name: str,
    seeds: list[int],
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_workers: int | None = None,
    game_record_db: Path | None = None,
    run_label: str = "",
) -> TwoTeamSelfPlayResult:
    """Runs `len(seeds)` two-team games, the same (spymaster, guesser)
    pair on both sides of each, across `max_workers` processes.

    `game_record_db`, if given, persists every game's board layout and
    turn sequence to that SQLite file (codenames/llm_store.py) under
    `run_label`, so it can be inspected later without replaying (see
    scripts/tools/dump_game_records.py) -- most useful when `guesser_name` costs
    real money per turn (e.g. "llm")."""
    stats = _new_stats()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(
            sims_cache_dir,
            spymaster_cls,
            spymaster_kwargs,
            guesser_pool_config,
            guesser_name,
            max_turns,
            game_record_db,
            run_label,
        ),
    ) as executor:
        for result in executor.map(_play_task, seeds):
            update_stats(stats, result)
    return finalize_result(stats)


# --------------------------------------------------------------------------
# Head-to-head: two DIFFERENT spymasters, one per side.
#
# Distinct from run_two_team_self_play above, which puts the same spymaster on
# both sides. The difference is not just plumbing: update_stats() pools every
# turn of a game into one accumulator, which is right when both sides are the
# same model and wrong when they are not, and TwoTeamSelfPlayResult has no
# win rate at all -- in self-play there is no one to win.
#
# Team A holds 9 words and moves first; team B holds 8. That is a real
# advantage, so a matchup is only interpretable if each board is played twice
# with the sides swapped. run_two_team_matchup does that by construction
# rather than leaving it to the caller.
# --------------------------------------------------------------------------


@dataclass
class SideStats:
    """One spymaster's record across a matchup, counting only the turns it
    actually took."""

    spymaster: str
    games: int = 0
    wins: int = 0
    wins_as_first: int = 0
    games_as_first: int = 0
    assassin_losses: int = 0  # games this side ended by revealing the assassin
    clues: int = 0
    clue_number_sum: int = 0
    correct_sum: int = 0  # own words revealed on this side's own clues
    guesses: int = 0
    guess_own: int = 0
    guess_opponent: int = 0
    guess_neutral: int = 0
    guess_assassin: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def assassin_rate(self) -> float:
        return self.assassin_losses / self.games if self.games else 0.0

    @property
    def mean_clue_number(self) -> float:
        return self.clue_number_sum / self.clues if self.clues else 0.0

    @property
    def mean_correct_per_clue(self) -> float:
        return self.correct_sum / self.clues if self.clues else 0.0

    @property
    def own_rate(self) -> float:
        return self.guess_own / self.guesses if self.guesses else 0.0


@dataclass
class MatchupResult:
    n_games: int
    n_boards: int
    timeouts: int
    sides: dict[str, SideStats]
    discarded_boards: tuple[int, ...] = ()
    """Seeds dropped because a guesser refused to rank on at least one of the
    board's two side assignments. Both assignments go, never one: a half-played
    board would still count toward win rate while silently vanishing from the
    paired sign test, which is the comparison that actually has power."""


def _update_side_stats(sides: dict[str, SideStats], result: TwoTeamGameResult, name_of: dict[str, str]) -> None:
    """`name_of` maps "A"/"B" to the spymaster that held that side in this
    game, which the swap makes vary game to game."""
    for team, name in name_of.items():
        sides[name].games += 1
    sides[name_of["A"]].games_as_first += 1

    if result.winner is not None:
        winner = name_of[result.winner]
        sides[winner].wins += 1
        if result.winner == "A":
            sides[winner].wins_as_first += 1
    if result.outcome == "loss" and result.turns:
        # play_two_team_game returns the instant the assassin is revealed,
        # so the last turn belongs to whoever revealed it.
        sides[name_of[result.turns[-1].team]].assassin_losses += 1

    for tt in result.turns:
        st = sides[name_of[tt.team]]
        st.clues += 1
        st.clue_number_sum += tt.turn.number
        st.correct_sum += sum(1 for _, role in tt.turn.guesses if role == Role.OWN)
        for _, role in tt.turn.guesses:
            st.guesses += 1
            setattr(st, f"guess_{role.value}", getattr(st, f"guess_{role.value}") + 1)


def _matchup_worker_init(
    sims_cache_dir: Path,
    spec_x: tuple,
    spec_y: tuple,
    names: tuple[str, str],
    guesser: str,
    guesser_pool_config: Path,
    max_turns: int,
    game_record_db: Path | None,
    run_label: str,
    vocabulary: list[str] | None = None,
    suite_id: str | None = None,
) -> None:
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["vocabulary"] = vocabulary
    _WORKER_STATE["suite_id"] = suite_id
    _WORKER_STATE["x"] = spec_x[0](**spec_x[1])
    _WORKER_STATE["y"] = spec_y[0](**spec_y[1])
    _WORKER_STATE["names"] = names
    _WORKER_STATE["guesser"] = build_guesser(guesser, guesser_pool_config)
    _WORKER_STATE["max_turns"] = max_turns
    _WORKER_STATE["record_store"] = GameRecordStore(game_record_db) if game_record_db is not None else None
    _WORKER_STATE["run_label"] = run_label


def _is_guesser_refusal(exc: Exception) -> bool:
    """An LLM guesser that will not produce a usable ranking raises rather than
    backfill board order (codenames/guessers/openai_compat.py), which is right
    -- a fabricated ranking would be cached and poison every later run. It is a
    property of the position, not a bug, so the board is discarded and the run
    continues."""
    return isinstance(exc, RuntimeError) and "usable ranking" in str(exc)


# Concurrent clue computations allowed per worker process under threads.
#
# One process cannot use more than ~2.5 cores of clue computation however many
# threads it runs -- about a third of a turn holds the GIL (measured scaling
# 1.0/1.9/2.3/2.6/2.5x at 1/4/8/16/32 threads) -- so letting more than a few
# compute at once buys no speed. It does cost memory: the Gaussian first stage
# allocates several hundred MB of temporaries per turn, and numpy releases the
# GIL inside those operations, so every thread can hold them at the same time.
# All games start on turn 1 together, and 2 processes x 16 unguarded threads
# peaked at 13 GB -- which at 6 processes is an OOM kill. Threads waiting on
# the guesser do not hold a slot, so this caps memory without capping the
# number of games in flight.
COMPUTE_SLOTS = 3


def _guard_spymasters(slots: int) -> None:
    """Wrap each spymaster's clue search in one per-process semaphore.

    Runs at the start of a chunk, before its threads exist, and once per
    process -- ProcessPoolExecutor hands a worker one chunk at a time, so there
    is nothing to race. This sets an attribute on the spymaster outside its
    __init__, which the thread-safety argument elsewhere says spymasters never
    do; it is safe here only because no thread is running yet.
    """
    if _WORKER_STATE.get("guarded"):
        return
    sem = threading.BoundedSemaphore(slots)
    for key in ("x", "y"):
        sm = _WORKER_STATE.get(key)
        inner = getattr(sm, "_score_all_clues", None)
        if inner is None:
            continue

        def guarded(*args, _inner=inner, **kwargs):
            with sem:
                return _inner(*args, **kwargs)

        sm._score_all_clues = guarded
    _WORKER_STATE["guarded"] = True


def _matchup_pull(tasks, threads: int) -> list[tuple]:
    """Keep `threads` games in flight in this process, pulling from a queue
    shared by every worker, until the queue is empty.

    Replaces fixed chunks. With games pre-assigned two chunks per process, a
    run was observed with five of six workers idle and one still playing the
    slowest games of its last chunk -- mostly guesser refusals, each four
    attempts at an 8k-token budget -- so the whole setting waited on one
    process. Pulling from a shared queue means no worker idles while games
    remain; the only tail left is the last games of the setting itself.

    Results carry their own (seed, swapped), so completion order is
    irrelevant to the caller. A game whose guesser refuses comes back as a
    discard; a genuine bug raises out of `f.result()` below and ends the run.
    """
    import queue as _queue

    _guard_spymasters(min(threads, COMPUTE_SLOTS))
    out: list[tuple] = []
    lock = threading.Lock()

    def loop() -> None:
        while True:
            try:
                task = tasks.get_nowait()
            except _queue.Empty:
                return
            result = _matchup_task(task)
            with lock:
                out.append(result)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for f in [ex.submit(loop) for _ in range(threads)]:
            f.result()
    return out


def _matchup_task(task: tuple[int, bool]) -> tuple[TwoTeamGameResult | None, bool, int, str | None]:
    seed, swapped = task
    state = _WORKER_STATE
    board = Board.generate(seed=seed, vocabulary=state.get("vocabulary"))
    guesser = state["guesser"]
    # Same listener on both sides -- the matchup is between spymasters, so
    # varying the guesser too would confound the comparison.
    first, second = (state["y"], state["x"]) if swapped else (state["x"], state["y"])
    name_a, name_b = (state["names"][1], state["names"][0]) if swapped else state["names"]

    by_role = board_by_role(board) if state["record_store"] is not None else None
    try:
        result = play_two_team_game(board, (first, guesser), (second, guesser), state["sims"], max_turns=state["max_turns"])
    except Exception as exc:                                   # noqa: BLE001 -- classified below
        # Anything raised here has to cross a process boundary, and not every
        # exception survives the trip. An openai APIStatusError does not:
        # __init__ demands keyword-only `response` and `body`, unpickling it
        # raises TypeError inside the pool's reader, and the run dies with
        # BrokenProcessPool naming neither the board nor the cause. A 15-setting
        # sweep was lost to exactly that. So nothing unpicklable leaves this
        # function -- errors become strings here, at the point they are raised.
        kind = type(exc).__module__.split(".")[0]
        transient = kind == "openai" or isinstance(exc, (ConnectionError, TimeoutError))
        if not (transient or _is_guesser_refusal(exc)):
            # A real bug still stops the run, but as a readable error rather
            # than a broken pool. The original type is kept in the message
            # because the type itself may not survive pickling.
            raise RuntimeError(
                f"{type(exc).__name__} in seed {seed}: {exc}\n"
                + "".join(traceback.format_exception(exc))
            ) from None
        return None, swapped, seed, f"{type(exc).__name__}: {exc}"
    if state["record_store"] is not None:
        # The side assignment is in the label because a recorded game
        # stores only "A"/"B" per turn -- without it a dumped transcript
        # can't say which spymaster gave which clue.
        # Under a frozen suite the game is also keyed by its seating and the
        # suite (codenames/eval_suite.py), so a rerun overwrites rather than
        # duplicates and only missing boards are ever played.
        suite_id = state.get("suite_id")
        state["record_store"].add_game(
            by_role, result, label=f"{state['run_label']}|A={name_a},B={name_b}",
            spymaster_id=f"A={name_a},B={name_b}" if suite_id else None, suite_id=suite_id)
    return result, swapped, seed, None


def run_two_team_matchup(
    spec_x: tuple,
    spec_y: tuple,
    names: tuple[str, str],
    guesser: str,
    seeds: list[int],
    guesser_pool_config: Path = DEFAULT_POOL_CONFIG,
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_workers: int | None = None,
    game_record_db: Path | None = None,
    run_label: str = "",
    progress: bool = False,
    threads_per_worker: int = 1,
    vocabulary: list[str] | None = None,
    suite_id: str | None = None,
) -> MatchupResult:
    """Plays each seed twice -- once with `names[0]` as team A, once with
    `names[1]` as team A -- so first-move advantage falls on both sides
    equally. Returns 2 * len(seeds) games. Both sides share one guesser,
    `guesser` -- a spec or pool name, see
    codenames/guessers/registry.py::build_guesser.

    `max_workers` is deliberately not defaulted to os.cpu_count(): with an
    LLM guesser the per-turn cost is network latency, not computation, so
    the useful worker count is bounded by how many games can be in flight
    at once, not by cores. A turn inside one game is strictly sequential
    (the board changes), so wall-clock is roughly
    (games / workers) * (turns per game) * latency.

    `vocabulary` restricts the boards to a word list (the frozen suite uses
    the held-out words); `suite_id` records every game under its suite and
    seating -- see codenames/eval_suite.py."""
    tasks = [(seed, swapped) for seed in seeds for swapped in (False, True)]
    sides = {names[0]: SideStats(names[0]), names[1]: SideStats(names[1])}
    if game_record_db is not None:
        # Create the file, set WAL, and build the tables once here, before
        # any worker exists. Otherwise every worker races to do it on a
        # db that may not exist yet, and the journal-mode switch needs an
        # exclusive lock (see codenames/llm_store.py::_connect).
        GameRecordStore(game_record_db)
    timeouts = 0
    done = 0

    # Processes x threads, because each alone wastes a different resource.
    #
    # A turn is ~1.8s of CPU followed by a guesser call that this endpoint has
    # served in anywhere from 2.6s to 30s, so throughput is bounded twice: by
    # games in flight / latency, and by CPU.
    #
    #   - Processes alone: each holds a private 1.13 GB (PSS, measured) copy of
    #     the tensor, booster and lexical tables, so a 30 GB box caps near 16
    #     and 48 once OOM-killed an editor. 16 in flight / 30s is 0.5 turns/s
    #     while the CPU sits mostly idle.
    #   - Threads alone: measured scaling on pure clue computation, no API, is
    #     1.0 / 1.9 / 2.3 / 2.6 / 2.5x at 1/4/8/16/32 threads -- about a third
    #     of a turn holds the GIL, so one process tops out near 1.5 turns/s
    #     however many threads it has.
    #   - Both: P processes of T threads each give P*T in flight and P GILs.
    #     6 x 16 is ~96 in flight and ~8.7 turns/s of CPU, which is 5-6x the
    #     16-process setup at either end of the latency range.
    #
    # An earlier note here said threads were MEASURED SLOWER (1.3 calls/min vs
    # 28). That measurement counted new rows in llm_store while the run was
    # replaying seeds an earlier run had already bought -- cache hits add no
    # rows -- so it measured cache state, not speed. The scaling numbers above
    # replace it.
    #
    # Threads within a process are safe for checked reasons: spymasters assign
    # no attributes outside __init__; SimilarityTensor is a read-only mmap;
    # llm_store opens WAL with check_same_thread=False; GameRecordStore writes
    # under _RECORD_LOCK. OMP_NUM_THREADS is forced to 1 for the workers, or
    # every thread's predict opens its own OpenMP pool (838 threads observed at
    # 100). Native-crash isolation is kept at process granularity: a segfault
    # loses one chunk, not the run.
    init_args = (sims_cache_dir, spec_x, spec_y, names, guesser, guesser_pool_config,
                 max_turns, game_record_db, run_label, vocabulary, suite_id)
    hybrid = threads_per_worker > 1
    saved_omp = os.environ.get("OMP_NUM_THREADS")
    manager = None
    if hybrid:
        # Spawned children inherit the environment at launch, and LightGBM's
        # OpenMP runtime reads it when it loads -- so it must be set here.
        os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_matchup_worker_init,
            initargs=init_args,
        ) as executor:
            if hybrid:
                # One shared queue, one puller per process; see _matchup_pull.
                manager = multiprocessing.get_context("spawn").Manager()
                shared = manager.Queue()
                for t in tasks:
                    shared.put(t)
                n_proc = max_workers or 1
                futures = [executor.submit(_matchup_pull, shared, threads_per_worker)
                           for _ in range(n_proc)]
                results = (r for f in futures for r in f.result())
            else:
                results = executor.map(_matchup_task, tasks)
            played, failed = [], {}
            for result, swapped, seed, err in results:
                if result is None:
                    failed.setdefault(seed, err)
                else:
                    played.append((result, swapped, seed))
                done += 1
                if progress and done % 10 == 0:
                    print(f"  {done}/{len(tasks)} games", flush=True)
    finally:
        if hybrid:
            if manager is not None:
                manager.shutdown()
            if saved_omp is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = saved_omp

    if failed:
        seed, err = next(iter(failed.items()))
        print(f"  discarded {len(failed)} board(s) -- a guesser would not rank. "
              f"first was seed {seed}: {err.splitlines()[0][:120]}", flush=True)
    for result, swapped, seed in played:
        if seed in failed:                    # drop this board's surviving half too
            continue
        name_of = {"A": names[1], "B": names[0]} if swapped else {"A": names[0], "B": names[1]}
        _update_side_stats(sides, result, name_of)
        if result.outcome == "timeout":
            timeouts += 1

    kept = [s_ for s_ in seeds if s_ not in failed]
    return MatchupResult(n_games=2 * len(kept), n_boards=len(kept), timeouts=timeouts,
                         sides=sides, discarded_boards=tuple(sorted(failed)))
