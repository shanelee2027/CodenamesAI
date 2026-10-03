"""Buy listener rankings for training the distilled guesser.

Generates board positions and clues, asks a cheap LLM to rank each, and lets
the response cache accumulate them. Designed to be started and forgotten:
leave it running while you work, kill it whenever, re-run the same command and
it resumes.

**Why generated positions rather than recorded games.** Every cached ranking
we already own came from a game where a spymaster chose a clue it believed was
good. So the distilled model has never seen a clue that is irrelevant to the
board, or one pointing at the opponent's words or the assassin. A spymaster's
search scores ~111k candidate clues per turn and nearly all of them are bad --
a listener that has only ever been shown good clues cannot tell it so. Hence
the `--mix`: deliberately adversarial and junk clues, not just plausible ones.

**Resumability is the cache, not a checkpoint file.** Every response is
written to cache/llm_store.db the moment it arrives
(codenames/llm_store.py), and the plan is a deterministic function of
`--seed`, so a re-run regenerates the identical position list and skips
everything already bought. Nothing is lost by a crash, a closed terminal, or
a machine that sleeps -- at worst the handful of calls in flight. There is no
partial state to corrupt because the unit of work is one cache row.

**Boards are built from the TRAINING wordlist**, not all 400 board words
(codenames/board.py::load_training_wordlist). The 150 held-out words exist so
evaluation uses board content no model was fitted on, and this model has
per-word features (`word_mean_sim`, `word_sd_sim`) that could memorise
word-specific behaviour. See docs/design-decisions.md, "Two structural guards
against overfitting".

**Clue pool is the same one the spymaster searches**: rarity percentile <= 10,
11,145 of 111,440 words, and legal against the board
(codenames/board.py::is_legal_clue). Sampling clues the spymaster could never
play would spend the budget teaching the model about words it will never see.

Usage:
    # ~10k positions, gentle on the machine, safe to leave running
    python scripts/data/collect_listener_data.py --n 10000

    # a test set on the held-out words, generated exactly like the rest
    python scripts/data/collect_listener_data.py --vocab holdout --n 2100

    # the same held-out test set ranked by Sonnet, stopping before $4 is spent
    python scripts/data/collect_listener_data.py --vocab holdout --n 2100 \
        --guesser anthropic:claude-sonnet-5:medium --budget 4.0

    # check how much is already bought without spending anything
    python scripts/data/collect_listener_data.py --n 10000 --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.board import Board, Role, is_legal_clue, load_holdout_wordlist, load_training_wordlist
from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

CACHE = PROJECT_ROOT / "cache"
DB = CACHE / "llm_store.db"

# Board seeds start far above anything the arenas use (0..~300), so generated
# training boards can never collide with a board an evaluation was run on.
SEED_BASE = 1_000_000

# `--vocab holdout`: the same generator on the 150 held-out words, for a TEST
# set only (docs/log.md, "A generated held-out-words test set"). The rankings
# from the eval suites' games are all on clues a spymaster chose, so they are
# far easier than generated positions and their R² cannot be compared with
# the generated "new boards" set. Positions made here differ from those only
# in vocabulary, so the gap between the two measures overfitting to the
# training words. Own seed range, far from every other source.
from codenames.listener_training import HOLDOUT_COLLECTED_BASE as HOLDOUT_SEED_BASE  # noqa: E402

# How clues are chosen. The point of the non-"own" rows is that a spymaster's
# search has to reject these, which it cannot learn to do from a training set
# containing only clues some spymaster already liked.
DEFAULT_MIX = {
    "own": 0.40,        # points at the team's own words -- the good case
    "opponent": 0.20,   # points at the other team's words
    "assassin": 0.10,   # points at the assassin
    "neutral": 0.10,    # points at neutrals
    "random": 0.20,     # drawn uniformly from the legal pool: usually junk
}

# Sample from the top of the similarity ranking rather than its argmax, so the
# same target word does not always produce the same clue.
TOP_POOL = 40


def build_clue_pool(stats: ClueStats, max_rarity: float) -> np.ndarray:
    return np.flatnonzero(stats.rarity_percentile <= max_rarity)


def clue_near(
    targets: list[str], sims, stats, pool: np.ndarray, space_i: int, rng: random.Random
) -> int | None:
    """A clue from the pool that scores highly against `targets` on average."""
    try:
        idxs = [sims.board_index[w.lower()] for w in targets]
    except KeyError:
        return None
    sim = np.asarray(sims.tensor[:, idxs, space_i], dtype=np.float32)[pool]  # (pool, len(targets))
    mean = stats.mean[pool, space_i][:, None]
    sd = stats.std[pool, space_i][:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((sim - mean) / sd).mean(axis=1)
    z = np.where(np.isfinite(z), z, -np.inf)
    top = np.argpartition(-z, min(TOP_POOL, len(z) - 1))[:TOP_POOL]
    return int(pool[rng.choice(top.tolist())])


def make_position(i: int, sims, stats, pool: np.ndarray, space_i: int, mix: dict, vocab: list[str],
                  seed_base: int = SEED_BASE):
    """One (clue, candidates, k) to buy. Deterministic in `i`."""
    rng = random.Random((seed_base + i) * 2654435761 % (2**63))
    board = Board.generate(seed=seed_base + i, vocabulary=vocab)

    # Reveal a random slice of the board, so positions span the whole arc of a
    # game rather than only fresh 25-word boards -- agreement with the teacher
    # varies strongly with how many words are left.
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
        targets = rng.sample(pool_words, min(len(pool_words), rng.randint(1, 3)))
        ci = clue_near(targets, sims, stats, pool, space_i, rng)
        if ci is None:
            return None

    clue = stats.clue_words[ci]
    if not is_legal_clue(clue, candidates):
        return None
    return {"clue": clue, "candidates": candidates, "k": rng.randint(1, 4), "kind": kind}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="positions to plan (already-cached ones cost nothing)")
    ap.add_argument("--start", type=int, default=0,
                    help="offset into the board-seed sequence: positions come from "
                         "SEED_BASE + start + i. A start beyond everything collected "
                         "so far yields boards this project has never trained, "
                         "validated or selected on, which is what an uncontaminated "
                         "test set requires -- re-splitting existing data cannot give "
                         "one, because the feature set and hyperparameters were chosen "
                         "against it.")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--provider", default="deepinfra")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--guesser", default=None,
                    help="a guesser spec (codenames/guessers/registry.py::build_guesser) used instead of "
                         "--model/--provider/--effort, e.g. anthropic:claude-sonnet-5:medium. The plan is "
                         "the same, so a second guesser ranks exactly the positions the first did")
    ap.add_argument("--budget", type=float, default=None,
                    help="Anthropic guessers only: stop starting calls once the billed tokens, priced at "
                         "--price-in/--price-out, reach this many dollars")
    ap.add_argument("--price-in", type=float, default=2.0, help="$ per million input tokens")
    ap.add_argument("--price-out", type=float, default=10.0,
                    help="$ per million output tokens, thinking included. The defaults are claude-sonnet-5's "
                         "as implied by this project's two measured Sonnet call costs (docs/log.md)")
    ap.add_argument("--max-workers", type=int, default=8,
                    help="kept low on purpose: these calls are network-bound and the provider "
                         "caps throughput anyway, so more workers only add 429s and CPU noise "
                         "while you are using the machine")
    ap.add_argument("--max-rarity", type=float, default=10.0, help="same clue pool the spymaster searches")
    ap.add_argument("--space", default="numberbatch")
    ap.add_argument("--vocab", choices=("training", "holdout"), default="training",
                    help="board words: the training list, or the held-out list for a test set only "
                         "(seeds from HOLDOUT_SEED_BASE)")
    ap.add_argument("--dry-run", action="store_true", help="plan and report coverage; buy nothing")
    ap.add_argument("--nice", type=int, default=10, help="process niceness, so this yields to your work")
    args = ap.parse_args()

    try:
        os.nice(args.nice)
    except (AttributeError, PermissionError):
        pass

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    pool = build_clue_pool(stats, args.max_rarity)
    space_i = sims.spaces.index(args.space)
    if args.vocab == "holdout":
        vocab, seed_base = load_holdout_wordlist(), HOLDOUT_SEED_BASE
    else:
        vocab, seed_base = load_training_wordlist(), SEED_BASE
    print(f"clue pool: {len(pool)} of {len(stats.clue_words)} at rarity<={args.max_rarity}")
    print(f"board vocabulary: {len(vocab)} {args.vocab} words, seeds from {seed_base + args.start}")

    print(f"planning {args.n} positions...", flush=True)
    positions, kinds = [], Counter()
    for i in range(args.n):
        p = make_position(args.start + i, sims, stats, pool, space_i, DEFAULT_MIX, vocab, seed_base)
        if p is not None:
            positions.append(p)
            kinds[p["kind"]] += 1
    print(f"planned {len(positions)}   mix: {dict(kinds)}")

    if args.guesser:
        from codenames.guessers.registry import build_guesser
        g = build_guesser(args.guesser)
    else:
        from codenames.guessers.openai_compat import OpenAICompatGuesser
        g = OpenAICompatGuesser(model=args.model, provider=args.provider,
                                reasoning_effort=args.effort, cache_path=DB)
    if args.budget is not None and not hasattr(g, "usage"):
        raise SystemExit("--budget needs a guesser that reports billed tokens (an anthropic: spec)")

    def spent() -> float:
        u = g.usage
        return (u["input_tokens"] * args.price_in + u["output_tokens"] * args.price_out) / 1e6
    cached = sum(1 for p in positions
                 if g._disk_cache.get(g.cache_model_id, p["clue"], tuple(p["candidates"]), p["k"]) is not None)
    todo = len(positions) - cached
    print(f"already cached: {cached}   to buy: {todo}"
          + ("" if args.guesser else f"   est ${todo * 0.000102:.2f}"))
    if args.dry_run or todo == 0:
        return

    done = threading.local()
    counter = {"n": 0, "err": 0}
    lock = threading.Lock()
    t0 = time.time()

    def work(p):
        if args.budget is not None and spent() >= args.budget:
            return
        try:
            g.rank_candidates(p["clue"], p["candidates"], None, number=p["k"])
        except Exception as exc:  # a dead host or a refusal must not kill an 8-hour run
            with lock:
                counter["err"] += 1
                if counter["err"] <= 5:
                    print(f"  error on {p['clue']!r}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        with lock:
            counter["n"] += 1
            n = counter["n"]
            if n % 100 == 0 or n == len(positions):
                el = time.time() - t0
                rate = n / el if el else 0
                print(f"  {n}/{len(positions)}  {rate:.2f}/s  errors {counter['err']}  "
                      f"eta {(len(positions)-n)/rate/60:.0f} min" if rate else f"  {n}",
                      f"  spent ${spent():.3f}" if args.budget is not None else "", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(work, positions))
    print(f"\ndone in {(time.time()-t0)/60:.1f} min, {counter['err']} errors. "
          f"Re-run the same command to top up or resume.")
    if hasattr(g, "usage"):
        u = g.usage
        print(f"billed: {u['calls']} calls, {u['input_tokens']} input / {u['output_tokens']} output tokens"
              + (f", ${spent():.3f} at ${args.price_in}/${args.price_out} per Mtok" if args.budget is not None else ""))


if __name__ == "__main__":
    main()
