"""Buy listener rankings on boards seeded with decoy words, to pin the scale.

The distilled listener's softmax normalises over the board, so the training
signal only ever says *which* of these words is best and never *whether any of
them is good*: add a constant to every score in a group and the likelihood is
unchanged. The absolute level is therefore not wrong so much as unsupervised.

Decoys fix that by fixing the composition of the comparison set. A board's
composition varies; words drawn uniformly from the vocabulary are a known,
constant reference, so "how much better than a random word is this?" becomes a
supervised quantity. That is the level, pinned to a fixed anchor.

**What the data says this is worth.** Measured on 272 probe boards at 15
decoys, the teacher puts a decoy first 19.5% of the time against a 37.5%
chance rate -- a ratio of 0.52, z = -6.1 -- and the effect tracks clue quality
(0.36 of chance for the listener's own top clue, 0.58 for the worst
shortlisted one). But the signal lives almost entirely in the FIRST pick:
positions 1 through 7 sit at 0.93-1.14 of chance, i.e. nothing. Rankings are
therefore truncated at and including the first decoy when fitting; below that
line own-rate is 33.2% against a 36% base rate, which is zero information.

**Why D is randomised rather than chosen.** More decoys means more chances
that one of them happens to relate to the clue and outranks the word the clue
actually meant. Subsampling the 15-decoy probe down confirms it: the ratio
degrades monotonically from 0.43 at D=1 to 0.52 at D=15. But that statistic is
feature-blind, and the model being fitted is not -- a decoy that wins because
it genuinely relates has similarity features saying so. So the curve cannot
settle D, and picking a compromise would just hide the assumption. Randomising
D over {2, 5, 10} instead makes the level's invariance to D a measurable
property: under Luce/IIA the fitted level is a property of the clue and must
not move with D. If it agrees across the three, contamination is not biasing
the fit and the larger D can be used for efficiency; if it drifts, the bias
has been detected rather than argued about. Randomising per position rather
than per batch keeps D unconfounded with anything that drifts over a run.

**The clue number is given, not withheld.** `k` is already a feature
(listener_features.py::extract, `number`), with -1 standing for "no number",
a regime the spymaster never operates in -- it always announces one. Data
collected without a number would land in that sentinel bucket and teach the
model about a task it never performs. Randomising k over 1-5 instead puts
support above the 2.2-2.5 the model typically announces, so the fit observes
positions past where the teacher stops knowing.

**Decoys are screened against the board, never against the clue.** A pool word
sharing a stem with a board word is not a decoy, it is a legitimate answer,
and `is_legal_clue` already has that check. Screening on *clue* relatedness
would be the tempting fix for the contamination above and is exactly wrong: it
makes the reference distribution clue-dependent, which destroys the fixed
anchor the whole design rests on, and it would have to be reproduced at
inference using the same embedding similarity the model is judged against.

**This goes through the ordinary response cache, deliberately.** The cache key
is (model, clue, candidates, number) and the decoys are part of `candidates`,
so these queries cannot collide with existing rows. That means the data can be
bought through `rank_candidates` under the real guesser prompt rather than a
bespoke one -- the same prompt the teacher sees in a game, so the rankings are
on-distribution. `scripts/tools/probe_anchors.py` had to bypass the cache
because it asked a different question with the same key; this does not.

Usage:
    python scripts/data/collect_decoy_data.py --n 2000 --out cache/training_data/decoys.jsonl
    python scripts/data/collect_decoy_data.py --n 2000 --out ... --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, Role, is_legal_clue, load_training_wordlist
from codenames.clue_stats import ClueStats
from codenames.guessers.registry import load_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

_spec = importlib.util.spec_from_file_location(
    "_collect", PROJECT_ROOT / "scripts" / "data" / "collect_listener_data.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# Disjoint from collect_listener_data's 1_000_000 so the two never generate the
# same board and silently duplicate a position under a different label.
SEED_BASE = 3_000_000
D_CHOICES = (2, 5, 10)
K_CHOICES = (1, 2, 3, 4, 5)


def make_position(i: int, sims, stats, pool: np.ndarray, space_i: int,
                  mix: dict, vocab: list[str]) -> dict | None:
    """One (clue, candidates+decoys, k) to buy. Deterministic in `i`."""
    rng = random.Random((SEED_BASE + i) * 2654435761 % (2**63))
    board = Board.generate(seed=SEED_BASE + i, vocabulary=vocab)

    words = list(board.words)
    n_reveal = rng.randint(0, max(0, len(words) - 5))
    revealed = set(rng.sample(words, n_reveal))
    candidates = [w for w in words if w not in revealed]
    if len(candidates) < 5:
        return None

    kinds = list(mix)
    kind = rng.choices(kinds, weights=[mix[k] for k in kinds])[0]
    role_for = {"own": Role.OWN, "opponent": Role.OPPONENT,
                "neutral": Role.NEUTRAL, "assassin": Role.ASSASSIN}
    if kind == "random":
        ci = int(rng.choice(pool.tolist()))
    else:
        pool_words = [w for w in board.words_by_role(role_for[kind]) if w in candidates]
        if not pool_words:
            return None
        n_target = min(len(pool_words), rng.randint(1, 3))
        ci = _base.clue_near(rng.sample(pool_words, n_target), sims, stats, pool, space_i, rng)
        if ci is None:
            return None
    clue = sims.clue_words[ci]
    if not is_legal_clue(clue, candidates):
        return None

    n_decoys = rng.choice(D_CHOICES)
    number = rng.choice(K_CHOICES)

    # Decoys are vocabulary words not on this board, screened against the board
    # only. A pool word sharing a stem with a board word is a legitimate answer,
    # not a decoy; relatedness to the CLUE is left alone on purpose.
    on_board = {w.lower() for w in words}
    off = [w for w in vocab if w.lower() not in on_board]
    decoys: list[str] = []
    for w in rng.sample(off, len(off)):
        if is_legal_clue(w, candidates):
            decoys.append(w.capitalize())
            if len(decoys) == n_decoys:
                break
    if len(decoys) < n_decoys:
        return None

    mixed = candidates + decoys
    rng.shuffle(mixed)
    return {"i": i, "seed": SEED_BASE + i, "clue": clue, "kind": kind,
            "number": number, "n_decoys": n_decoys, "n_board": len(candidates),
            "candidates": mixed, "decoys": decoys, "board_words": candidates}


def truncate(ranking: list[str], decoys: list[str]) -> tuple[int, bool]:
    """`(cut, saw_decoy)` -- board words before the first decoy, and whether one
    appeared at all. The fit uses ranking terms up to and INCLUDING index `cut`.

    A decoy the teacher leaves out of its ranking has not gone missing; it has
    been placed below everything it did rank, which is the strongest rejection
    available. Omitted decoys therefore sit at the bottom and do not create a
    cut.
    """
    dset = set(decoys)
    for j, w in enumerate(ranking):
        if w in dset:
            return j, True
    return len(ranking), False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    ap.add_argument("--space", default="numberbatch")
    ap.add_argument("--pool-config", type=Path,
                    default=PROJECT_ROOT / "configs" / "guesser_pool_oss120b.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--nice", type=int, default=10)
    args = ap.parse_args()

    try:
        os.nice(args.nice)
    except (AttributeError, OSError):
        pass

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    pool = _base.build_clue_pool(stats, args.max_rarity)
    space_i = sims.spaces.index(args.space)
    vocab = load_training_wordlist()

    positions = [p for p in (make_position(args.start + i, sims, stats, pool, space_i,
                                           _base.DEFAULT_MIX, vocab)
                             for i in range(args.n)) if p]
    kinds = {}
    for p in positions:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"planned {len(positions)} of {args.n}   mix: {kinds}")
    print(f"decoys: {sorted({p['n_decoys'] for p in positions})}   "
          f"numbers: {sorted({p['number'] for p in positions})}")
    if args.dry_run:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
        print(f"resuming: {len(done)} already written")
    positions = [p for p in positions if p["i"] not in done]

    guesser = load_pool(args.pool_config)["llm"].guesser
    lock = threading.Lock()
    fh = args.out.open("a")
    state = {"n": 0, "cuts": 0, "t0": time.time()}

    def work(p: dict) -> None:
        try:
            ranking = guesser.rank_candidates(p["clue"], p["candidates"], sims,
                                              number=p["number"])
        except Exception as exc:                                   # noqa: BLE001
            with lock:
                print(f"  seed {p['seed']}: {type(exc).__name__}: {exc}", flush=True)
            return
        cut, saw = truncate(ranking, p["decoys"])
        row = dict(p, ranking=ranking, cut=cut, saw_decoy=saw)
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            state["n"] += 1
            state["cuts"] += saw
            if state["n"] % 25 == 0:
                el = time.time() - state["t0"]
                print(f"  {state['n']}/{len(positions)}  decoy seen in "
                      f"{state['cuts']}/{state['n']}  {el/state['n']:.2f}s/pos", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(work, positions))
    fh.close()
    print(f"\n{state['n']} positions -> {args.out}")


if __name__ == "__main__":
    main()
