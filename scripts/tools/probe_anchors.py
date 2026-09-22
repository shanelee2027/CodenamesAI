"""Do PASS and decoy anchors carry signal about clue quality?

Both ideas try to give the listener an absolute notion of "the guesser has no
idea", which the current softmax cannot express: it normalises over board words
only, so a clue is scored on the ranking it induces and never on how
confidently it induces it. A clue at z=1 over four own words and one at z=3
over the same four are indistinguishable to it.

    PASS    the teacher inserts a sentinel into its ranking at the point it
            stops recognising a connection. A true outside option, which pins
            the scale -- but it depends on gpt-oss having reliable access to
            its own uncertainty, which LLMs frequently do not.

    DECOYS  words drawn from the board vocabulary that are NOT on this board
            are mixed into the candidate list. A board word the teacher ranks
            below a decoy is one it sees no more in than a word the clue was
            never chosen for. No introspection is asked of the model, and the
            anchor is scored in the same units as everything else, so it is
            immune to the scale question rather than answering it.

Neither is worth a re-collection if the anchor does not MOVE with clue quality,
which is what this measures. Three clue grades per board, from the project's
own machinery so "good" and "bad" are not hand-picked:

    top     the learned listener's own choice -- a clue it stands behind
    mid     ~300th by the Gaussian first stage -- plausible but middling,
            which is exactly the case under suspicion
    random  a random admissible clue -- the floor

If PASS lands late for `top` and early for `random`, it is real. If decoys sink
below every board word for `top` and rise above some for `random`, that is
real. If either is flat across all three grades, that anchor is inert and the
re-collection it would cost is wasted.

Calls the API directly rather than through LLMGuesser: these are new prompts,
and the response cache is keyed on (model, clue, candidates, number) with no
prompt in it, so routing them through the cache would poison it with answers to
a different question.

Usage:
    python scripts/tools/probe_anchors.py --boards 40 --out probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, Role, is_legal_clue
from codenames.guessers.registry import load_pool
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

N_DECOYS = 5
SENTINEL = "PASS"

# The clue number is deliberately withheld. Telling the model the spymaster
# meant 4 would anchor the confidence boundary on 4, and the probe would
# measure the prompt instead of the clue.
PASS_PROMPT = """You are playing the guesser role in the board game Codenames. \
Your spymaster gave the clue "{clue}". Here are the words still available to guess:

{words}

Rank ALL of these words from most to least related to the clue. Then insert the \
single token "PASS" into your ranking at the point where you stop recognising any \
real connection to the clue: everything above PASS is a word you can see a link \
to, everything below it is a word you cannot. PASS may come first (no word is \
clearly linked) or last (every word is).

Judge only whether you see a connection. Do not consider whether guessing would \
be risky.

Respond with ONLY a JSON array of the words plus "PASS", e.g. ["word1", "word2", \
"PASS", "word3"] -- no other text."""

PLAIN_PROMPT = """You are playing the guesser role in the board game Codenames. \
Your spymaster gave the clue "{clue}". Here are the words still available to guess:

{words}

Rank ALL of these words from most to least related to the clue.

Respond with ONLY a JSON array of the words -- no other text."""


def parse_array(text: str, allowed: set[str]) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out, seen = [], set()
    for w in raw:
        if isinstance(w, str) and w in allowed and w not in seen:
            out.append(w)
            seen.add(w)
    return out


def ask(guesser, prompt: str, attempts: int = 4) -> str:
    """Retry on the degenerate repetition loop and on transient API errors --
    unattended runs cannot afford to stop for either."""
    last = ""
    for i in range(attempts):
        req = {"model": guesser.model, "max_completion_tokens": guesser.max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
        if guesser.reasoning_effort:
            req["reasoning_effort"] = guesser.reasoning_effort
        if i:
            req["frequency_penalty"] = 1.0
        try:
            last = guesser.client.chat.completions.create(**req).choices[0].message.content or ""
        except Exception as exc:                                  # noqa: BLE001
            last = ""
            time.sleep(2 * (i + 1))
            continue
        if last.strip():
            return last
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=40)
    ap.add_argument("--decoys", type=int, default=N_DECOYS)
    ap.add_argument("--shortlist-sample", type=int, default=0,
                    help="replace the three grades with K clues drawn from the listener's own\n                         top-50. The three-grade contrast is confounded: decoy wins are common\n                         on random clues and random clues fail anyway, so pooling grades\n                         conflates the anchor with the grade. Within the shortlist -- the only\n                         clues the deployed model ever plays -- that confound is gone.")
    ap.add_argument("--skip-pass", action="store_true",
                    help="PASS measured inert (r=-0.068, p=0.47); skip to spend the calls on decoys")
    ap.add_argument("--first-seed", type=int, default=90000)
    ap.add_argument("--mid-rank", type=int, default=300, help="Gaussian rank for the 'mid' clue")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(12345)
    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    sm = LearnedListenerSpymaster()
    guesser = load_pool(PROJECT_ROOT / "configs" / "guesser_pool_oss120b.json")["llm"].guesser
    vocab = list(sims.board_index)                       # board vocabulary, lowercased

    admissible = np.flatnonzero(
        (sm.clue_stats.rarity_percentile <= sm.max_rarity)
        & (~sm.acronym_mask if sm.acronym_mask is not None else True))

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for b in range(args.boards):
        seed = args.first_seed + b
        board = Board.generate(seed=seed)
        words = list(board.words)
        ctx = TurnContext(board=board, turn_index=0)

        # three clue grades, from the project's own scoring rather than taste
        top = sm.top_clues(ctx, sims, 1)[0][0]
        _, g_scores, _ = sm._first_stage._score_all_clues(board, sims)
        finite = np.flatnonzero(np.isfinite(g_scores))
        order = finite[np.argsort(-g_scores[finite])]
        mid = next((sims.clue_words[int(c)] for c in order[args.mid_rank:args.mid_rank + 40]
                    if is_legal_clue(sims.clue_words[int(c)], words)), None)
        rand = next((sims.clue_words[int(c)] for c in rng.sample(list(admissible), 60)
                     if is_legal_clue(sims.clue_words[int(c)], words)), None)
        if mid is None or rand is None:
            continue

        # decoys: real board-vocabulary words NOT on this board, so they match
        # the candidates in register and frequency and differ only in never
        # having been selected for by this clue
        off = [w for w in vocab if w.lower() not in {x.lower() for x in words}]
        decoys = [w.capitalize() for w in rng.sample(off, args.decoys)]

        if args.shortlist_sample:
            pool = [sims.clue_words[int(c)] for c in order[:50]
                    if is_legal_clue(sims.clue_words[int(c)], words)]
            picks = rng.sample(pool, min(args.shortlist_sample, len(pool)))
            grades = [(f"shortlist{j}", c) for j, c in enumerate(picks)]
        else:
            grades = [("top", top), ("mid", mid), ("random", rand)]
        for grade, clue in grades:
            # --- PASS ---
            shuffled = words[:]
            rng.shuffle(shuffled)
            if args.skip_pass:
                rank, pass_pos = [], None
            else:
                txt = ask(guesser, PASS_PROMPT.format(clue=clue, words="\n".join(shuffled)))
                rank = parse_array(txt, set(shuffled) | {SENTINEL})
                pass_pos = rank.index(SENTINEL) / max(1, len(words)) if SENTINEL in rank else None

            # --- DECOYS ---
            mixed = shuffled + decoys
            rng.shuffle(mixed)
            txt2 = ask(guesser, PLAIN_PROMPT.format(clue=clue, words="\n".join(mixed)))
            rank2 = parse_array(txt2, set(mixed))
            dset = set(decoys)
            # A decoy the teacher leaves OUT of its ranking has not gone
            # missing -- it has been placed below everything it did rank, which
            # is the strongest rejection available. Scoring that as absent
            # would throw away the cleanest observations and bias the estimate
            # toward positions where the model was unsure enough to include
            # them. Omitted decoys therefore sit at the bottom.
            n_decoys_ranked = sum(1 for w in rank2 if w in dset)
            if rank2:
                positions = [i for i, w in enumerate(rank2) if w in dset]
                cut = min(positions) if positions else len(rank2)
                top_decoy = cut / max(1, len(rank2))
                # board words the teacher rates BELOW a word the clue never chose
                below = sum(1 for i, w in enumerate(rank2) if w not in dset and i > cut)
            else:
                top_decoy, below = None, None

            rows.append({"seed": seed, "grade": grade, "clue": clue,
                         "n_words": len(words), "pass_pos": pass_pos,
                         "pass_parsed": SENTINEL in rank, "top_decoy": top_decoy,
                         "board_below_decoy": below, "n_ranked": len(rank2),
                         "n_decoys_ranked": n_decoys_ranked, "n_decoys": args.decoys,
                         # The averages were inert and the tail was not: a decoy
                         # competes with the ~21 unrelated board words, so its
                         # typical rank tracks the size of that soup rather than
                         # the clue. Only the top of the list discriminates, so
                         # that is where the resolution goes.
                         "decoys_in_top3": sum(1 for w in rank2[:3] if w in dset),
                         "decoys_in_top5": sum(1 for w in rank2[:5] if w in dset),
                         "decoy_wins": bool(rank2 and rank2[0] in dset),
                         "ranking": rank2, "decoys": decoys})
            args.out.write_text(json.dumps(rows, indent=1))      # incremental
        print(f"  board {b+1}/{args.boards} (seed {seed}) done", flush=True)

    print(f"\n{len(rows)} observations -> {args.out}")


if __name__ == "__main__":
    main()
