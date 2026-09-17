"""Worker helpers for scratch/clue_number_arc.ipynb.

Lives in a module, not the notebook, because ProcessPoolExecutor under
"spawn" pickles the task function by qualified name -- functions defined in
a notebook cell have no importable module to resolve against.

The question: how does the announced clue number fall as a team's own words
run out, and does that depend on sigma? Randomly sampling board states
cannot answer it. The words left late in a real game are not a random
subset -- they are the ones that RESISTED being clued, because the easy
ones went first. `shuffled_twin` below is the control that measures exactly
that gap.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# gain_and_penalty allocates ~68MB transients ((n_cand, GRID_CELLS, n_words)
# float32) several times per clue search. glibc serves blocks that size with
# mmap and munmaps them on free, so 16 workers churning them generate a
# page-fault storm: measured sys time at 0.93x user (293s of kernel against
# 315s of work). Keeping them on the heap for reuse drops that to 0.04x.
# Wall-clock gain is only ~10% -- the pool is memory-bandwidth bound, not
# fault bound -- but it stops burning half the machine on page faults.
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "1073741824")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "1073741824")

import random
from pathlib import Path

from codenames.board import Board, Card, Role
from codenames.guessers.registry import load_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.expected_words import ExpectedWordsSpymaster

DEFAULT_GUESSER = "noisy_numberbatch"
POOL_CONFIG = Path("configs/guesser_pool.json")

_W: dict = {}


def init(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    import torch

    torch.set_num_threads(1)
    _W["sims"] = SimilarityTensor.load(cache_dir)
    _W["sm"] = {}
    _W["g"] = {}


def _sm(sigma: float) -> ExpectedWordsSpymaster:
    if sigma not in _W["sm"]:
        _W["sm"][sigma] = ExpectedWordsSpymaster(space="numberbatch", sigma=sigma, max_rarity=10.0)
    return _W["sm"][sigma]


def _g(name: str):
    if name not in _W["g"]:
        _W["g"][name] = load_pool(POOL_CONFIG)[name].guesser
    return _W["g"][name]


def play_arc(task):
    """(sigma, seed, guesser_name) -> one row per turn, playing until the
    team's own words are exhausted.

    **Misses never end the game.** A neutral, opponent or assassin reveal
    ends the *turn* exactly as in real Codenames, but play continues --
    that is the modification that lets one game trace the whole arc from 9
    own words down to 1 instead of stopping at the first assassin.
    """
    sigma, seed, guesser_name = task
    sims, sm, guesser = _W["sims"], _sm(sigma), _g(guesser_name)
    board = Board.generate(seed=seed)
    rows = []
    for _ in range(40):  # generous cap; a stuck game would otherwise spin
        own_left = board.remaining(Role.OWN)
        if own_left == 0:
            break
        # Snapshot BEFORE the reveals: this is the state the clue was given
        # in, and what the shuffled twin must reproduce.
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
        rows.append({
            "sigma": sigma, "seed": seed, "own_before": own_left, "k": number,
            "got": got, "stopped": stopped, "clue": clue,
            "revealed_by_role": revealed_by_role,
        })
    return rows


def shuffled_twin(task):
    """(sigma, seed, revealed_by_role) -> the clue number on a board with the
    SAME words and the same number revealed per role, but randomly chosen
    ones. The matched control for 'the easy words went first'.
    """
    sigma, seed, revealed_by_role, twin_seed = task
    sims = _W["sims"]
    board = Board.generate(seed=seed)
    rng = random.Random(twin_seed)
    for role_value, count in revealed_by_role.items():
        pool = [c.word for c in board.cards if c.role.value == role_value]
        for w in rng.sample(pool, min(count, len(pool))):
            board.reveal(w)
    if board.remaining(Role.OWN) == 0:
        return None
    ctx = TurnContext(board=board, turn_index=len(board.revealed))
    _, number = _sm(sigma).give_clue(ctx, sims)
    return {"sigma": sigma, "seed": seed, "own_before": board.remaining(Role.OWN), "k": number}
