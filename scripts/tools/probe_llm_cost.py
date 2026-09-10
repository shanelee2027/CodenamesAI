"""Measure what an LLM guesser turn actually costs, on real board positions.

Every cost number in docs/log.md before this script was an estimate built
from character counts, because the visible JSON array is the only part of
the response that can be measured offline. It is also the *smaller* part:
with thinking on, `usage.output_tokens` covers deliberation as well, and
deliberation is what the bill is made of. This script settles the split by
asking, which costs a few cents.

It answers three questions at once:

1. What is the thinking/text split? Inferred as
   `output_tokens - (tokens in the visible text)`, the visible part
   measured with the free `count_tokens` endpoint rather than a
   chars/token guess.
2. What would 50 games cost? Projected from the measured per-call usage
   times the measured call count (`CALLS_PER_50_GAMES` below, from
   simulating 50 two-team games offline).
3. Does the live path work at all? The eval pipeline has only ever run
   against mocks. The last stage makes one real `LLMGuesser` call so that
   response parsing, the disk cache write, and the cache read-back are
   exercised against the real API before a paid run depends on them.

Variants compared: response format (full ranking vs. top-n only) crossed
with whatever efforts are passed. The format question is whether cutting
the array to the n words the guesser would name is worth losing the full
ranking -- the ranking being the only way to see *where* the assassin sat
when a clue went wrong.

Nothing here writes to the eval store, so a probe run cannot contaminate a
suite's results. The one real LLMGuesser call does populate
cache/llm_store.db, which is intended: it is a genuine response, keyed by
model and effort like any other.

Usage:
    python scripts/tools/probe_llm_cost.py --dry-run          # free; show the prompts
    python scripts/tools/probe_llm_cost.py -n 5               # ~$0.05
    python scripts/tools/probe_llm_cost.py -n 5 --effort low --effort medium
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, load_holdout_wordlist
from codenames.env import load_env
from codenames.guessers.llm import _PROMPT_TEMPLATE, LLMGuesser
from codenames.similarity import SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.registry import spymaster_spec

# Measured by simulating 50 two-team games with the baseline spymaster on
# both sides: mean 14.3 guesser calls per game. Regenerate rather than
# trusting this constant if the spymaster's mean clue number moves much,
# since a larger clue number ends games in fewer turns.
CALLS_PER_50_GAMES = 715

# Only used to turn measured tokens into dollars. VERIFY against the
# current pricing page before quoting anything downstream -- these are
# inputs to the report, not facts the script can check.
DEFAULT_RATES = {  # model prefix -> (input $/Mtok, output $/Mtok)
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
}

# The proposed alternative format. Kept here rather than in
# codenames/guessers/llm.py because it is a candidate being priced, not
# the format the guesser uses -- promoting it is a separate decision that
# changes what every cached response means.
_TOPN_TEMPLATE = """You are playing the guesser role in the board game Codenames. Your \
spymaster gave the clue "{clue}"{count_note}. Here are the words still available to guess:

{words}

Name the {n} word(s) you would guess for this clue, best first. Respond with ONLY a JSON \
array of the words as strings, e.g. ["word1", "word2"] -- no other text."""


@dataclass
class Turn:
    """One real position: a board, the baseline's clue for it, and the
    words a guesser would be choosing among."""

    seed: int
    clue: str
    number: int
    candidates: list[str]

    def prompt(self, fmt: str) -> str:
        count_note = f" for {self.number} word(s)" if self.number else ""
        words = "\n".join(self.candidates)
        if fmt == "full":
            return _PROMPT_TEMPLATE.format(clue=self.clue, count_note=count_note, words=words)
        return _TOPN_TEMPLATE.format(
            clue=self.clue, count_note=count_note, words=words, n=self.number
        )


def collect_turns(n: int, sims: SimilarityTensor) -> list[Turn]:
    """Real positions, not synthetic ones: the prompt's token count
    depends on how many words are still unrevealed, so measuring on fresh
    boards alone would overstate every call after the opening."""
    cls, kwargs = spymaster_spec("expected_words")
    spymaster = cls(**kwargs)
    vocab = load_holdout_wordlist()
    turns = []
    # Reveal a few more words on each successive board so the sample spans
    # opening and mid-game positions rather than 25-word boards only.
    for i in range(n):
        board = Board.generate(2000 + i, vocabulary=vocab)
        rng = random.Random(2000 + i)
        for w in rng.sample(board.words, min(i * 2, 12)):
            board.reveal(w)
        ctx = TurnContext(board=board, turn_index=len(board.revealed))
        clue, number, _ = spymaster.top_clues(ctx, sims, 1)[0]
        turns.append(
            Turn(
                seed=board.seed,
                clue=clue,
                number=number,
                candidates=[w for w in board.words if not board.is_revealed(w)],
            )
        )
    return turns


def rate_for(model: str) -> tuple[float, float]:
    for prefix, rates in DEFAULT_RATES.items():
        if model.startswith(prefix):
            return rates
    return (0.0, 0.0)


def measure(client, model: str, turns: list[Turn], fmt: str, effort: str | None, max_tokens: int):
    """One request per turn; returns per-call usage plus the inferred
    thinking split."""
    rows = []
    for t in turns:
        prompt = t.prompt(fmt)
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effort is not None:
            request["output_config"] = {"effort": effort}
        else:
            request["thinking"] = {"type": "disabled"}
        resp = client.messages.create(**request)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        # count_tokens is free, so the visible-text size is measured
        # rather than estimated; thinking is then what's left over.
        visible = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text or "-"}]
        ).input_tokens
        rows.append(
            {
                "seed": t.seed,
                "clue": t.clue,
                "number": t.number,
                "n_candidates": len(t.candidates),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "visible_tokens": visible,
                "thinking_tokens": max(0, resp.usage.output_tokens - visible),
                "stop_reason": resp.stop_reason,
                "text": text,
            }
        )
    return rows


def report(label: str, rows: list[dict], model: str) -> dict:
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    in_tok, out_tok = mean("input_tokens"), mean("output_tokens")
    vis, think = mean("visible_tokens"), mean("thinking_tokens")
    r_in, r_out = rate_for(model)
    cost = CALLS_PER_50_GAMES * (in_tok * r_in + out_tok * r_out) / 1_000_000
    truncated = sum(1 for r in rows if r["stop_reason"] == "max_tokens")
    print(
        f"{label:<24} in {in_tok:>6.0f}  out {out_tok:>6.0f}  "
        f"(visible {vis:>5.0f} + thinking {think:>6.0f})   "
        f"50 games ~ ${cost:>7.2f}"
        + (f"   [{truncated}/{n} hit max_tokens]" if truncated else "")
    )
    return {"label": label, "input": in_tok, "output": out_tok, "visible": vis,
            "thinking": think, "cost_50_games": cost, "truncated": truncated}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--calls", type=int, default=5,
                    help="positions per variant (default 5)")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", action="append", default=None,
                    help="repeatable; omit for the thinking-disabled path")
    ap.add_argument("--format", action="append", choices=["full", "topn"], default=None,
                    help="repeatable (default: both)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="matches LLMGuesser's default (default 512)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompts and the planned spend, call nothing")
    ap.add_argument("--out", type=Path, default=None, help="write raw rows as JSON")
    args = ap.parse_args()

    efforts = args.effort if args.effort else [None]
    formats = args.format if args.format else ["full", "topn"]

    sims = SimilarityTensor.load()
    turns = collect_turns(args.calls, sims)
    n_requests = len(turns) * len(efforts) * len(formats)

    print(f"model {args.model}   positions {len(turns)}   "
          f"formats {formats}   efforts {efforts}")
    print(f"requests: {n_requests}   (projection base: {CALLS_PER_50_GAMES} calls for 50 games)\n")
    for t in turns:
        print(f"  seed {t.seed}: {t.clue.upper()} {t.number}   "
              f"{len(t.candidates)} candidates unrevealed")

    if args.dry_run:
        print("\n--- example prompts ---")
        for fmt in formats:
            print(f"\n[{fmt}]\n{turns[0].prompt(fmt)}")
        print(f"\nDry run: no requests sent. Drop --dry-run to send {n_requests}.")
        return

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set; nothing was sent. Put it in .env "
            "(see .env.example) or export it for one command."
        )

    import anthropic

    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else None
    )

    print("\n--- measured, per call ---")
    all_rows, summaries = {}, []
    for fmt in formats:
        for effort in efforts:
            label = f"{fmt}/{effort or 'no-thinking'}"
            rows = measure(client, args.model, turns, fmt, effort, args.max_tokens)
            all_rows[label] = rows
            summaries.append(report(label, rows, args.model))

    print(f"\n(rates assumed: ${rate_for(args.model)[0]}/${rate_for(args.model)[1]} per Mtok "
          f"input/output -- VERIFY before quoting)")

    if len(summaries) > 1:
        base = summaries[0]
        print("\n--- vs. the first variant ---")
        for s in summaries[1:]:
            d = base["cost_50_games"] - s["cost_50_games"]
            pct = 100 * d / base["cost_50_games"] if base["cost_50_games"] else 0
            print(f"  {s['label']:<24} saves ${d:>6.2f}  ({pct:>5.1f}%)")

    print("\n--- example response ---")
    first = next(iter(all_rows.values()))[0]
    print(f"  {first['clue'].upper()} {first['number']} -> {first['text'][:200]}")

    # Smoke test: the eval path has only ever run against mocks. This
    # exercises the real thing end to end -- parse, disk-cache write, then
    # read back -- so a paid run doesn't discover a broken cache halfway.
    print("\n--- LLMGuesser smoke test (real parse + disk cache) ---")
    t = turns[0]
    guesser = LLMGuesser(model=args.model, effort=efforts[0],
                         cache_path=PROJECT_ROOT / "cache" / "llm_store.db")
    ranking = guesser.rank_candidates(t.clue, t.candidates, sims, number=t.number)
    ok_complete = sorted(ranking) == sorted(t.candidates)
    fresh = LLMGuesser(model=args.model, effort=efforts[0],
                       cache_path=PROJECT_ROOT / "cache" / "llm_store.db")
    cached = fresh.rank_candidates(t.clue, t.candidates, sims, number=t.number)
    print(f"  ranked {len(ranking)} words; every candidate present: {ok_complete}")
    print(f"  disk cache returns the same ranking: {cached == ranking}")
    print(f"  top 3: {ranking[:3]}")

    if args.out:
        args.out.write_text(json.dumps(all_rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
