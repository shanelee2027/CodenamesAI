"""Two-team self-play arena: bulk-runs codenames.game.play_two_team_game
with the SAME spymaster+guesser pair on both sides, across many seeded
boards, in parallel worker processes -- mirrors codenames/arena.py's
process-parallel structure, but for one symmetric pair rather than a
spymaster x guesser cross-product.

With both teams running identical logic, *which* team wins isn't
informative -- it's mostly just the 9-vs-8 first-move edge every game
already has, not a signal about model quality (see docs/log.md). The
question this answers instead is the same one the single-team arena
already answers -- how often does this spymaster/guesser combination's
own play end in an assassin hit, versus a clean finish -- just measured
in a real two-team game where the board depletes from *both* sides'
actual play, not the single-team framing's static distractors. Both
teams' turns are pooled into one set of stats (they're the same model),
mirroring codenames/arena.py's CrossPlayResult fields where the concepts
carry over, but keyed by "clean finish vs. assassin ending" instead of
"win vs. loss," since a genuine win rate here is a coin flip modulated by
the first-move edge, not a quality signal.

`guesser_name` can also be `MIXED_GUESSER` ("mixed"): instead of fixing
one guesser for the whole run, each game independently draws one,
uniformly, from every guesser in `guesser_pool_config` -- matching the
distribution the spymaster was actually trained against, rather than
the narrower test a single fixed
guesser is.
"""

from __future__ import annotations

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from codenames.board import Board, Role
from codenames.game import DEFAULT_MAX_TURNS, TwoTeamGameResult, play_two_team_game
from codenames.guessers.registry import load_pool, training_pool
from codenames.llm_store import GameRecordStore, board_by_role
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

# Passed as `guesser_name` to mean "don't fix one guesser -- each game
# independently draws one, uniformly, from every guesser in
# --guesser-pool-config" (drawing from configs/guesser_pool.json, the
# source of truth for pool
# composition -- evaluating against a single fixed guesser instead is a
# narrower test than what the model was actually trained against).
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
    # Constructs the spymaster fresh inside the worker (not by pickling
    # an existing instance across the process boundary) -- same reason
    # codenames/arena.py does this: avoids pickling issues and, combined
    # with "spawn" below, sidesteps the CUDA-after-fork hazard documented
    # there.
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["spymaster"] = spymaster_cls(**spymaster_kwargs)
    _WORKER_STATE["guesser_name"] = guesser_name
    if guesser_name == MIXED_GUESSER:
        _WORKER_STATE["guesser_pool"] = list(training_pool(guesser_pool_config).values())
    else:
        _WORKER_STATE["guesser"] = load_pool(guesser_pool_config)[guesser_name].guesser
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
    guesser_pool_config: Path,
    guesser_name: str,
    max_turns: int,
    game_record_db: Path | None,
    run_label: str,
) -> None:
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["x"] = spec_x[0](**spec_x[1])
    _WORKER_STATE["y"] = spec_y[0](**spec_y[1])
    _WORKER_STATE["names"] = names
    _WORKER_STATE["guesser"] = load_pool(guesser_pool_config)[guesser_name].guesser
    _WORKER_STATE["max_turns"] = max_turns
    _WORKER_STATE["record_store"] = GameRecordStore(game_record_db) if game_record_db is not None else None
    _WORKER_STATE["run_label"] = run_label


def _matchup_task(task: tuple[int, bool]) -> tuple[TwoTeamGameResult | None, bool, int, str | None]:
    seed, swapped = task
    state = _WORKER_STATE
    board = Board.generate(seed=seed)
    guesser = state["guesser"]
    # Same listener on both sides -- the matchup is between spymasters, so
    # varying the guesser too would confound the comparison.
    first, second = (state["y"], state["x"]) if swapped else (state["x"], state["y"])
    name_a, name_b = (state["names"][1], state["names"][0]) if swapped else state["names"]

    by_role = board_by_role(board) if state["record_store"] is not None else None
    try:
        result = play_two_team_game(board, (first, guesser), (second, guesser), state["sims"], max_turns=state["max_turns"])
    except RuntimeError as exc:
        # An LLM guesser that will not produce a usable ranking raises rather
        # than backfill board order (codenames/guessers/openai_compat.py), which
        # is right -- a fabricated ranking would be cached and poison every later
        # run. But it arrives here as an exception inside a pool worker, and
        # letting it propagate ends the whole matchup: a single refusal on one
        # board of one setting cost an 11-setting sweep 7 hours of work. Report
        # it as a discard and let the caller drop the board.
        return None, swapped, seed, str(exc)
    if state["record_store"] is not None:
        # The side assignment is in the label because a recorded game
        # stores only "A"/"B" per turn -- without it a dumped transcript
        # can't say which spymaster gave which clue.
        state["record_store"].add_game(by_role, result, label=f"{state['run_label']}|A={name_a},B={name_b}")
    return result, swapped, seed, None


def run_two_team_matchup(
    spec_x: tuple,
    spec_y: tuple,
    names: tuple[str, str],
    guesser_pool_config: Path,
    guesser_name: str,
    seeds: list[int],
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_workers: int | None = None,
    game_record_db: Path | None = None,
    run_label: str = "",
    progress: bool = False,
) -> MatchupResult:
    """Plays each seed twice -- once with `names[0]` as team A, once with
    `names[1]` as team A -- so first-move advantage falls on both sides
    equally. Returns 2 * len(seeds) games.

    `max_workers` is deliberately not defaulted to os.cpu_count(): with an
    LLM guesser the per-turn cost is network latency, not computation, so
    the useful worker count is bounded by how many games can be in flight
    at once, not by cores. A turn inside one game is strictly sequential
    (the board changes), so wall-clock is roughly
    (games / workers) * (turns per game) * latency."""
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

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_matchup_worker_init,
        initargs=(sims_cache_dir, spec_x, spec_y, names, guesser_pool_config, guesser_name, max_turns, game_record_db, run_label),
    ) as executor:
        played, failed = [], {}
        for result, swapped, seed, err in executor.map(_matchup_task, tasks):
            if result is None:
                failed.setdefault(seed, err)
            else:
                played.append((result, swapped, seed))
            done += 1
            if progress and done % 10 == 0:
                print(f"  {done}/{len(tasks)} games", flush=True)

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
