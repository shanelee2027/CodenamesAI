"""Two-team self-play arena: bulk-runs codenames.game.play_two_team_game
with the SAME codemaster+guesser pair on both sides, across many seeded
boards, in parallel worker processes -- mirrors codenames/arena.py's
process-parallel structure, but for one symmetric pair rather than a
codemaster x guesser cross-product.

With both teams running identical logic, *which* team wins isn't
informative -- it's mostly just the 9-vs-8 first-move edge every game
already has, not a signal about model quality (see docs/log.md). The
question this answers instead is the same one the single-team arena
already answers -- how often does this codemaster/guesser combination's
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
distribution the codemaster was actually trained against (see
docs/design-decisions.md), rather than the narrower test a single fixed
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
# --guesser-pool-config" (matching training's own sampling: see
# scripts/generate_training_data.py's `rng.choice(guesser_names)` and
# docs/design-decisions.md's "guesser pool is 3 members, equally
# weighted" -- evaluating against a single fixed guesser instead is a
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
    codemaster_cls: type,
    codemaster_kwargs: dict,
    guesser_pool_config: Path,
    guesser_name: str,
    max_turns: int,
    game_record_db: Path | None,
    run_label: str,
) -> None:
    # Constructs the codemaster fresh inside the worker (not by pickling
    # an existing instance across the process boundary) -- same reason
    # codenames/arena.py does this: avoids pickling issues and, combined
    # with "spawn" below, sidesteps the CUDA-after-fork hazard documented
    # there.
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["codemaster"] = codemaster_cls(**codemaster_kwargs)
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
    team = (state["codemaster"], guesser)
    # Snapshotted before any word is revealed -- play_two_team_game
    # mutates this same Board object in place.
    by_role = board_by_role(board) if state["record_store"] is not None else None
    result = play_two_team_game(board, team, team, state["sims"], max_turns=state["max_turns"])
    if state["record_store"] is not None:
        state["record_store"].add_game(by_role, result, label=state["run_label"])
    return result


def run_two_team_self_play(
    codemaster_cls: type,
    codemaster_kwargs: dict,
    guesser_pool_config: Path,
    guesser_name: str,
    seeds: list[int],
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_workers: int | None = None,
    game_record_db: Path | None = None,
    run_label: str = "",
) -> TwoTeamSelfPlayResult:
    """Runs `len(seeds)` two-team games, the same (codemaster, guesser)
    pair on both sides of each, across `max_workers` processes.

    `game_record_db`, if given, persists every game's board layout and
    turn sequence to that SQLite file (codenames/llm_store.py) under
    `run_label`, so it can be inspected later without replaying (see
    scripts/dump_game_records.py) -- most useful when `guesser_name` costs
    real money per turn (e.g. "llm")."""
    stats = _new_stats()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(
            sims_cache_dir,
            codemaster_cls,
            codemaster_kwargs,
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
