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
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from codenames.board import Board
from codenames.game import DEFAULT_MAX_TURNS, TwoTeamGameResult, play_two_team_game
from codenames.guessers.registry import load_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor


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
    )


_WORKER_STATE: dict = {}


def _worker_init(
    sims_cache_dir: Path,
    codemaster_cls: type,
    codemaster_kwargs: dict,
    guesser_pool_config: Path,
    guesser_name: str,
    max_turns: int,
) -> None:
    # Constructs the codemaster fresh inside the worker (not by pickling
    # an existing instance across the process boundary) -- same reason
    # codenames/arena.py does this: avoids pickling issues and, combined
    # with "spawn" below, sidesteps the CUDA-after-fork hazard documented
    # there.
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["codemaster"] = codemaster_cls(**codemaster_kwargs)
    _WORKER_STATE["guesser"] = load_pool(guesser_pool_config)[guesser_name].guesser
    _WORKER_STATE["max_turns"] = max_turns


def _play_task(seed: int) -> TwoTeamGameResult:
    state = _WORKER_STATE
    board = Board.generate(seed=seed)
    team = (state["codemaster"], state["guesser"])
    return play_two_team_game(board, team, team, state["sims"], max_turns=state["max_turns"])


def run_two_team_self_play(
    codemaster_cls: type,
    codemaster_kwargs: dict,
    guesser_pool_config: Path,
    guesser_name: str,
    seeds: list[int],
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_workers: int | None = None,
) -> TwoTeamSelfPlayResult:
    """Runs `len(seeds)` two-team games, the same (codemaster, guesser)
    pair on both sides of each, across `max_workers` processes."""
    stats = _new_stats()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(sims_cache_dir, codemaster_cls, codemaster_kwargs, guesser_pool_config, guesser_name, max_turns),
    ) as executor:
        for result in executor.map(_play_task, seeds):
            update_stats(stats, result)
    return finalize_result(stats)
