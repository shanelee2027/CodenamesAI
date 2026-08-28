"""Cross-play evaluation (SCOPE.md §M6): every codemaster x every guesser,
over a fixed set of seeded boards.

"Off-diagonal results are the ones that matter" (SCOPE §M6) -- a codemaster
that only does well against guessers it was implicitly tuned around has
overfit, which is why this always plays the *full* guesser pool (including
the held-out members), not just the training-visible ones. The held-out
flag is carried through into the results/DB purely as a label for later
filtering, once M8's learned codemaster makes the training/held-out split
actually matter for who gets trained against what.

Multiprocessing note (SCOPE §7's memory design note): each worker process
opens its own read-only mmap over the same similarity_tensor.npy via
SimilarityTensor.load() -- the OS page cache shares the underlying physical
pages across processes, so the ~1-2GB tensor itself is not duplicated per
worker. Codemasters/guessers must not defeat this by materializing their own
private copy of the full tensor as an instance cache -- see
codenames/codemasters/linear_scorer.py's docstring for a case where an
earlier version did exactly that and pushed worker RSS past 9GB.
"""

from __future__ import annotations

import json
import os
import resource
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from codenames.board import Board, Role
from codenames.codemasters.base import Codemaster
from codenames.game import DEFAULT_MAX_TURNS, GameResult, play_game
from codenames.guessers import load_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

CodemasterSpec = tuple[type[Codemaster], dict]


@dataclass
class CrossPlayResult:
    codemaster: str
    guesser: str
    held_out: bool
    n_games: int
    win_rate: float
    assassin_rate: float
    mean_turns: float
    mean_own_words_per_clue: float


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codemaster TEXT NOT NULL,
            guesser TEXT NOT NULL,
            guesser_held_out INTEGER NOT NULL,
            board_seed INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            clue TEXT NOT NULL,
            number INTEGER NOT NULL,
            guesses_json TEXT NOT NULL,
            reward REAL NOT NULL,
            ended_reason TEXT NOT NULL,
            game_outcome TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _log_game(conn: sqlite3.Connection, cm_name: str, g_name: str, held_out: bool, result: GameResult) -> None:
    rows = [
        (
            cm_name,
            g_name,
            int(held_out),
            result.seed,
            turn_index,
            turn.clue,
            turn.number,
            json.dumps([(w, role.value) for w, role in turn.guesses]),
            turn.reward,
            turn.ended_reason,
            result.outcome,
        )
        for turn_index, turn in enumerate(result.turns)
    ]
    conn.executemany(
        """
        INSERT INTO turns
            (codemaster, guesser, guesser_held_out, board_seed, turn_index,
             clue, number, guesses_json, reward, ended_reason, game_outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


_WORKER_STATE: dict = {}


def _worker_init(sims_cache_dir: Path, codemaster_specs: dict[str, CodemasterSpec], guesser_pool_config: Path, max_turns: int) -> None:
    _WORKER_STATE["sims"] = SimilarityTensor.load(sims_cache_dir)
    _WORKER_STATE["codemasters"] = {name: cls(**kwargs) for name, (cls, kwargs) in codemaster_specs.items()}
    _WORKER_STATE["guesser_pool"] = load_pool(guesser_pool_config)
    _WORKER_STATE["max_turns"] = max_turns


def _play_task(task: tuple[str, str, int]) -> tuple[str, str, int, GameResult, int]:
    cm_name, g_name, seed = task
    state = _WORKER_STATE
    board = Board.generate(seed=seed)
    codemaster = state["codemasters"][cm_name]
    guesser = state["guesser_pool"][g_name].guesser
    result = play_game(board, codemaster, guesser, state["sims"], max_turns=state["max_turns"])
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return cm_name, g_name, os.getpid(), result, rss_kb


def run_arena(
    codemaster_specs: dict[str, CodemasterSpec],
    guesser_pool_config: Path,
    seeds: list[int],
    db_path: Path,
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_workers: int | None = None,
) -> tuple[dict[tuple[str, str], CrossPlayResult], dict[int, int]]:
    """Play every codemaster against every guesser over `seeds`. Returns
    (results keyed by (codemaster, guesser), per-worker-pid max RSS in KB)."""
    guesser_held_out = {name: entry.held_out for name, entry in load_pool(guesser_pool_config).items()}
    tasks = [
        (cm_name, g_name, seed) for cm_name in codemaster_specs for g_name in guesser_held_out for seed in seeds
    ]

    conn = _init_db(db_path)
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "turns": 0, "turn_count": 0, "own_words": 0}
    )
    worker_rss: dict[int, int] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(sims_cache_dir, codemaster_specs, guesser_pool_config, max_turns),
    ) as pool:
        for cm_name, g_name, pid, result, rss_kb in pool.map(_play_task, tasks):
            held_out = guesser_held_out[g_name]
            _log_game(conn, cm_name, g_name, held_out, result)

            s = stats[(cm_name, g_name)]
            s["games"] += 1
            s["wins"] += result.outcome == "win"
            s["losses"] += result.outcome == "loss"
            s["turns"] += len(result.turns)
            for turn in result.turns:
                s["turn_count"] += 1
                s["own_words"] += sum(1 for _, role in turn.guesses if role == Role.OWN)

            worker_rss[pid] = max(worker_rss.get(pid, 0), rss_kb)

    conn.commit()
    conn.close()

    results = {
        (cm_name, g_name): CrossPlayResult(
            codemaster=cm_name,
            guesser=g_name,
            held_out=guesser_held_out[g_name],
            n_games=int(s["games"]),
            win_rate=s["wins"] / s["games"],
            assassin_rate=s["losses"] / s["games"],
            mean_turns=s["turns"] / s["games"],
            mean_own_words_per_clue=(s["own_words"] / s["turn_count"]) if s["turn_count"] else 0.0,
        )
        for (cm_name, g_name), s in stats.items()
    }
    return results, worker_rss
