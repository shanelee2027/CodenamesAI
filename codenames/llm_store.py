"""Persistent SQLite-backed storage for real-money LLM eval runs (see
docs/log.md): API responses, so a crash or accidental rerun never re-pays
for an identical (model, clue, candidates, number) query, and full game
records (board layout + turn sequence), so a human can inspect exactly
what happened in a real run without replaying it -- see
scripts/dump_game_records.py for the human-readable view.

Both live in the same on-disk file (default cache/llm_store.db,
gitignored like the rest of cache/) so the arena's multiprocessing
workers only need one WAL-mode SQLite connection each, rather than two
separate storage mechanisms to reason about. WAL mode is what makes
concurrent writers *across processes* (the CPU arena's workers) safe
without any extra locking here. Within one process, a single
sqlite3.Connection isn't safe to use from multiple threads at once (the
GPU-batched arena now plays many games' LLMGuesser calls concurrently on
a thread pool -- see two_team_gpu_arena.py) -- LLMResponseCache guards
that with its own lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from codenames.board import Role

if TYPE_CHECKING:
    # Deferred: codenames/guessers/llm.py (which depends on this module)
    # is imported by codenames/guessers/__init__.py, which codenames/game.py
    # itself imports -- a real import here would close that into a
    # circular import. `from __future__ import annotations` already makes
    # every annotation in this file a lazy string, so this is only ever
    # needed by type checkers.
    from codenames.game import TwoTeamGameResult

DEFAULT_DB_PATH = Path("cache/llm_store.db")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: callers that touch this connection from more
    # than one thread (LLMResponseCache, from LLMGuesser instances shared
    # across a GPU-batched arena's thread pool) are responsible for their
    # own locking around it -- see LLMResponseCache's `_lock`.
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS responses (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            clue TEXT NOT NULL,
            candidates TEXT NOT NULL,
            number INTEGER,
            ranking TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS game_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed INTEGER NOT NULL,
            label TEXT,
            board TEXT NOT NULL,
            turns TEXT NOT NULL,
            outcome TEXT NOT NULL,
            winner TEXT,
            total_reward TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def board_by_role(board) -> dict[Role, list[str]]:
    """Snapshot a board's word layout by role -- must be called before any
    word is revealed (play mutates the same Board object in place)."""
    return {role: board.words_by_role(role) for role in (Role.OWN, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN)}


class LLMResponseCache:
    """Write-through disk cache for LLMGuesser's API calls, keyed by the
    exact (model, clue, candidates, number) tuple -- the same key
    LLMGuesser's in-memory `_cache` already uses, just durable across
    process restarts and shared across the arena's worker processes."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self._conn = _connect(db_path)
        self._lock = threading.Lock()

    @staticmethod
    def _key(model: str, clue: str, candidates: tuple[str, ...], number: int | None) -> str:
        return json.dumps([model, clue, list(candidates), number], sort_keys=True)

    def get(self, model: str, clue: str, candidates: tuple[str, ...], number: int | None) -> list[str] | None:
        key = self._key(model, clue, candidates, number)
        with self._lock:
            row = self._conn.execute("SELECT ranking FROM responses WHERE cache_key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, clue: str, candidates: tuple[str, ...], number: int | None, ranking: list[str]) -> None:
        key = self._key(model, clue, candidates, number)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (cache_key, model, clue, candidates, number, ranking) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, model, clue, json.dumps(list(candidates)), number, json.dumps(ranking)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class GameRecordStore:
    """Durable, human-inspectable record of a real two-team game: the
    board layout by role (captured before any reveals, from team A's
    perspective -- Role.OWN is A's words, Role.OPPONENT is B's) and the
    full turn-by-turn sequence. Written once per game, so a run's
    transcripts survive after the process exits and never need to be
    replayed (re-paying for API calls) to inspect."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self._conn = _connect(db_path)

    def add_game(self, by_role: dict[Role, list[str]], result: TwoTeamGameResult, label: str = "") -> None:
        board_json = json.dumps({role.value: words for role, words in by_role.items()})
        turns_json = json.dumps(
            [
                {
                    "team": tt.team,
                    "clue": tt.turn.clue,
                    "number": tt.turn.number,
                    "guesses": [[w, role.value] for w, role in tt.turn.guesses],
                    "ended_reason": tt.turn.ended_reason,
                }
                for tt in result.turns
            ]
        )
        self._conn.execute(
            "INSERT INTO game_records (seed, label, board, turns, outcome, winner, total_reward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                result.seed,
                label,
                board_json,
                turns_json,
                result.outcome,
                result.winner,
                json.dumps(result.total_reward),
            ),
        )
        self._conn.commit()

    def all_games(self, label: str | None = None) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        if label is None:
            rows = self._conn.execute("SELECT * FROM game_records ORDER BY id").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM game_records WHERE label = ? ORDER BY id", (label,)).fetchall()
        self._conn.row_factory = None
        return rows

    def close(self) -> None:
        self._conn.close()
