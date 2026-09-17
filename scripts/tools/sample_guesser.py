"""Show how one or more guesser models read real clues, side by side.

Qualitative counterpart to scripts/tools/compare_guesser_models.py, which
scores models numerically. This prints the board and the ranking each model
returned, so you can read *why* a model would have taken the words it took --
the thing a yield number hides.

Positions come from games already recorded in cache/llm_store.db, so they are
real clues on real boards, not synthetic ones. Anthropic models are usually
already cached from earlier runs and cost nothing to re-show.

Usage:
    python scripts/tools/sample_guesser.py -n 5 \
        --model deepinfra:openai/gpt-oss-120b \
        --model claude-sonnet-5:medium
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Role
from codenames.guessers.llm import LLMGuesser
from codenames.guessers.openai_compat import OpenAICompatGuesser

CACHE = PROJECT_ROOT / "cache" / "llm_store.db"
MARK = {"own": "O", "opponent": "x", "neutral": ".", "assassin": "!!"}


def build(spec: str):
    """'claude-sonnet-5:medium' -> LLMGuesser; 'deepinfra:openai/gpt-oss-120b'
    -> OpenAICompatGuesser. Split on the FIRST colon; open-weight model ids
    contain slashes but not colons."""
    head, _, tail = spec.partition(":")
    if head.startswith("claude"):
        effort = None if tail in ("", "none") else tail
        return spec, LLMGuesser(model=head, effort=effort, cache_path=CACHE)
    return spec, OpenAICompatGuesser(model=tail, provider=head, cache_path=CACHE)


def positions(n: int, seed: int):
    """Real (board, clue, number, candidates) turns from recorded games."""
    conn = sqlite3.connect(CACHE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT seed, label, board, turns FROM game_records").fetchall()
    conn.close()
    out = []
    for r in rows:
        by_role = json.loads(r["board"])
        role_of = {w: role for role, ws in by_role.items() for w in ws}
        revealed: list[str] = []
        for t in json.loads(r["turns"]):
            # Shuffled, NOT left in `role_of` order. The recorded board is
            # stored grouped by role, so iterating it yields every own word,
            # then every opponent word, then neutrals -- which would both
            # leak the answer to the model as block structure in the prompt,
            # and make any board-order fallback score a perfect 4/4 for team
            # A and 0/4 for team B. Real play passes board order
            # (codenames/game.py), which is role-shuffled at construction
            # (codenames/board.py). The original order isn't recoverable from
            # the record, so a per-position shuffle stands in for it: seeded
            # by (seed, clue) so re-running shows the same position twice.
            cands = [w for w in role_of if w not in revealed]
            random.Random(f"{r['seed']}|{t['clue']}").shuffle(cands)
            if t["clue"] and len(cands) > 2:
                out.append({"label": r["label"], "seed": r["seed"], "clue": t["clue"],
                            "number": t["number"], "team": t["team"], "cands": cands,
                            "role_of": role_of, "played": [w for w, _ in t["guesses"]]})
            revealed.extend(w for w, _ in t["guesses"])
    return random.Random(seed).sample(out, min(n, len(out)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--positions", type=int, default=5)
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable. 'claude-sonnet-5:medium' or 'deepinfra:openai/gpt-oss-120b'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=6, help="ranking entries to show per model")
    args = ap.parse_args()

    models = [build(s) for s in args.model]
    for i, p in enumerate(positions(args.positions, args.seed), 1):
        # Team B's own words are stored as "opponent" from team A's view.
        ours = "opponent" if p["team"] == "B" else "own"
        intended = [w for w in p["cands"] if p["role_of"][w] == ours]
        print("=" * 78)
        print(f"[{i}] seed {p['seed']}  team {p['team']}  clue {p['clue']!r} for {p['number']}"
              f"   ({p['label'].split('|')[0]})")
        print(f"    their words : {', '.join(intended)}")
        print(f"    distractors : {', '.join(w for w in p['cands'] if p['role_of'][w] != ours)}")
        print(f"    actually played: {' -> '.join(p['played']) or '(none)'}")
        for spec, g in models:
            try:
                ranking = g.rank_candidates(p["clue"], p["cands"], None, number=p["number"])
            except Exception as exc:  # a missing key or a dead host shouldn't kill the run
                print(f"    {spec:34} ERROR {type(exc).__name__}: {exc}")
                continue
            shown = []
            for w in ranking[: args.top]:
                role = p["role_of"][w]
                shown.append(f"{w}({MARK['own' if role == ours else role if role != 'own' else 'opponent']})")
            asn = next((j for j, w in enumerate(ranking) if p["role_of"][w] == "assassin"), None)
            would = []
            for w in ranking[: p["number"]]:
                would.append(w)
                if p["role_of"][w] != ours:
                    break
            got = sum(1 for w in would if p["role_of"][w] == ours)
            print(f"    {spec:34} {' '.join(shown)}")
            print(f"    {'':34} -> takes {got}/{p['number']}, assassin at rank {asn + 1 if asn is not None else '?'}")
    print("=" * 78)
    print("O = their word   x = opponent's   . = neutral   !! = assassin")


if __name__ == "__main__":
    main()
