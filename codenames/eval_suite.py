"""Frozen eval suite + eval store (docs/iteration-architecture.md steps
5-6): evaluation against the LLM guesser is real money and should be
paid for exactly once per (matchup, suite, board, seating).

**The suite.** Fixed board seeds, built from
`codenames.board.load_holdout_wordlist()` (150 of 400 words, none of
which any model has trained on -- see docs/design-decisions.md's
held-out-board-words note), against one fixed LLM guesser. The LLM model id
is recorded as part of `EvalSuite.suite_id` -- switching models invalidates
every cached response and makes old numbers non-comparable (see
docs/iteration-architecture.md's "two roles for guessers"), so it must
invalidate the suite identity too. `board_seeds` is deliberately *not* part
of `suite_id`: extending a suite from 100 to 300 boards should reuse every
already-recorded game under the old 100, not treat the suite as a new one.

**What is played.** A head-to-head matchup, the same comparison every sweep
makes (codenames/two_team_arena.py::run_two_team_matchup): a challenger
against an opponent, every board played twice with the sides swapped so the
first-move advantage falls on both equally. The result is paired by board
(codenames/headtohead.py), and the sign test on boards one side swept is the
comparison with power.

**The store.** `codenames.llm_store.GameRecordStore` writes one row per
game, keyed here by (seating, suite_id, board_seed), where the seating is
`"A=<id>,B=<id>"`. Each seating of a board is therefore its own row: a rerun
overwrites rather than duplicates, and `run_eval_suite` plays only the
boards missing either seating.

**Model identity.** `spymaster_identity` hashes a spymaster's parameters and
the *content* of any model file it loads, never a path -- so retraining a
booster in place, or changing one parameter, is a different model, and the
same bytes are the same model wherever they live.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from codenames.board import load_holdout_wordlist
from codenames.headtohead import PairedSummary, SideTurns, paired_summary, side_turn_stats
from codenames.llm_store import DEFAULT_DB_PATH, GameRecordStore
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.two_team_arena import run_two_team_matchup

DEFAULT_EVAL_SUITE_CONFIG = Path(__file__).parent.parent / "configs" / "eval_suite.json"


@dataclass(frozen=True)
class EvalSuite:
    name: str
    board_seeds: tuple[int, ...]
    guesser: str
    """A guesser spec (codenames/guessers/registry.py::build_guesser), e.g.
    `anthropic:claude-sonnet-5:medium`: provider, model and effort, which
    together are also the guesser's response-cache identity."""

    @property
    def suite_id(self) -> str:
        """Identity independent of `board_seeds` on purpose -- see this
        module's docstring. Depends on the guesser spec, so a different
        model or effort is a different suite rather than a silent reuse of
        results bought from another listener."""
        payload = {"name": self.name, "guesser": self.guesser}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_eval_suite(config: Path = DEFAULT_EVAL_SUITE_CONFIG) -> EvalSuite:
    parsed = json.loads(Path(config).read_text())
    return EvalSuite(name=parsed["name"], board_seeds=tuple(parsed["board_seeds"]), guesser=parsed["guesser"])


def file_content_hash(path: Path) -> str:
    """The sha256 of a model file's raw bytes, not its path."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def spymaster_identity(name: str, params: dict | None = None, files: Iterable[Path] = ()) -> str:
    """A spymaster's identity for the eval store: its registry name, plus a
    hash of its constructor parameters and of the content of every model file
    it loads. A bare name when there is nothing to hash.

    Parameters are hashed as passed, so a change to a class *default* is not
    seen -- pin anything that matters in the params, as
    scripts/tools/sweep_role_costs.py does with MODEL_PINS."""
    files = sorted(str(f) for f in files)
    if not params and not files:
        return name
    h = hashlib.sha256(json.dumps(params or {}, sort_keys=True, default=str).encode())
    for f in files:
        h.update(file_content_hash(Path(f)).encode())
    return f"{name}:{h.hexdigest()[:12]}"


def seating(id_a: str, id_b: str) -> str:
    """The store key for one side assignment: who sat as team A and team B."""
    return f"A={id_a},B={id_b}"


def missing_seeds(ids: tuple[str, str], suite: EvalSuite, game_record_db: Path = DEFAULT_DB_PATH) -> list[int]:
    """Suite boards not yet recorded in BOTH seatings for this pair."""
    store = GameRecordStore(game_record_db)
    try:
        done = (store.recorded_seeds(seating(*ids), suite.suite_id)
                & store.recorded_seeds(seating(ids[1], ids[0]), suite.suite_id))
    finally:
        store.close()
    return [s for s in suite.board_seeds if s not in done]


@dataclass
class EvalResult:
    ids: tuple[str, str]
    paired: PairedSummary
    turns: dict[str, SideTurns]
    played: int
    discarded: tuple[int, ...]


def summarise_suite(ids: tuple[str, str], suite: EvalSuite, game_record_db: Path = DEFAULT_DB_PATH) -> EvalResult:
    """Everything recorded for this pair on this suite's boards, in both
    seatings, read back from the store alone."""
    con = sqlite3.connect(f"file:{game_record_db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT label, seed, winner, turns FROM game_records "
            "WHERE suite_id = ? AND spymaster_id IN (?, ?)",
            (suite.suite_id, seating(*ids), seating(ids[1], ids[0])),
        ).fetchall()
    finally:
        con.close()
    seeds = set(suite.board_seeds)
    rows = [r for r in rows if r[1] in seeds]
    paired = paired_summary(rows)
    per_seed = Counter(r[1] for r in rows)
    turns = side_turn_stats([r for r in rows if per_seed[r[1]] == 2])
    return EvalResult(ids=ids, paired=paired, turns=turns, played=0, discarded=())


def run_eval_suite(
    spec_x: tuple,
    spec_y: tuple,
    ids: tuple[str, str],
    suite: EvalSuite,
    game_record_db: Path = DEFAULT_DB_PATH,
    sims_cache_dir: Path = DEFAULT_CACHE_DIR,
    max_workers: int | None = None,
    threads_per_worker: int = 1,
    progress: bool = False,
) -> EvalResult:
    """Play every suite board not yet recorded in both seatings for this
    pair, then summarise everything recorded for it.

    A rerun of a fully evaluated pair plays nothing and makes no LLM calls.
    Extending `suite.board_seeds` plays only the new boards. A board the
    guesser refused to rank is discarded by the matchup and simply stays
    missing, so the next run retries it -- at the cost of only the calls that
    were refused, since everything else is in the response cache."""
    missing = missing_seeds(ids, suite, game_record_db)
    discarded: tuple[int, ...] = ()
    if missing:
        result = run_two_team_matchup(
            spec_x, spec_y, ids,
            guesser=suite.guesser,
            seeds=missing,
            sims_cache_dir=sims_cache_dir,
            max_workers=max_workers,
            game_record_db=game_record_db,
            run_label=f"eval:{suite.name}",
            progress=progress,
            threads_per_worker=threads_per_worker,
            vocabulary=load_holdout_wordlist(),
            suite_id=suite.suite_id,
        )
        discarded = result.discarded_boards
    out = summarise_suite(ids, suite, game_record_db)
    out.played = len(missing) - len(discarded)
    out.discarded = discarded
    return out
