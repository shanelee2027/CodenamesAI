"""How often does pick_temperature_listener choose a different clue or number
than learned_listener? Free: no API calls.

Positions are real game states: every half-turn of the recorded games in the
given files (scripts/pipeline/train_win_actor_critic.py play; both sides),
rebuilt as (board seed, revealed words, side to move). Both spymasters are
built from configs/spymasters.json with the same overrides, so the only
difference between them is the per-pick temperatures.

    python scripts/tools/compare_pick_temperature.py cache/training_data/win_games_val_incumbent_nonumber.jsonl
    python scripts/tools/compare_pick_temperature.py ... --param max_number=None     # the uncapped pair
    python scripts/tools/compare_pick_temperature.py ... --models conceptnet_listener within_turn_listener

`--models` compares any two registry entries (default: the pair above).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_W: dict = {}


MODELS = ["learned_listener", "pick_temperature_listener"]


def _init(params: dict, models: list[str]) -> None:
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    _W["sims"] = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    _W["models"] = models
    for name in models:
        cls, kw = spymaster_spec(name)
        _W[name] = cls(**{**kw, **params})


def _run(pos: tuple[int, list[int], str, int]) -> dict:
    from codenames.spymasters.base import TurnContext
    from codenames.win_game import make_board, view_for

    seed, revealed, mover, turn = pos
    board = view_for(make_board(seed, set(revealed)), mover)
    ctx = TurnContext(board=board, turn_index=turn)
    out = {"turn": turn, "revealed": len(revealed)}
    for name in _W["models"]:
        clue, k, _ = _W[name].top_clues(ctx, _W["sims"], 1)[0]
        out[name] = (clue, k)
    return out


def parse_value(v: str):
    if v.lower() == "none":
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+", type=Path)
    ap.add_argument("--param", action="append", default=[], help="k=v override for both spymasters")
    ap.add_argument("--n", type=int, default=600, help="positions to compare (first n)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--models", nargs=2, default=MODELS)
    args = ap.parse_args()
    params = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.param)}

    positions = []
    for path in args.games:
        for line in path.read_text().splitlines():
            g = json.loads(line)
            if g.get("error"):
                continue
            for t, h in enumerate(g["turns"]):
                positions.append((g["seed"], h["revealed"], h["mover"], t))
    positions = positions[: args.n]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(params, args.models)) as ex:
        rows = list(ex.map(_run, positions, chunksize=4))

    inc, new = args.models
    same_clue = sum(r[inc][0] == r[new][0] for r in rows)
    same_both = sum(r[inc] == r[new] for r in rows)
    print(f"{len(rows)} positions from {len(args.games)} file(s), overrides {params or 'none'}")
    print(f"  same clue and number: {same_both / len(rows):.1%}")
    print(f"  same clue:            {same_clue / len(rows):.1%}")
    print(f"  mean number: {inc} {sum(r[inc][1] for r in rows) / len(rows):.2f}, "
          f"{new} {sum(r[new][1] for r in rows) / len(rows):.2f}")
    ks = sorted({r[m][1] for r in rows for m in (inc, new)})
    ci, cn = Counter(r[inc][1] for r in rows), Counter(r[new][1] for r in rows)
    print("  number:            " + "  ".join(f"{k:>5d}" for k in ks))
    print(f"  {inc[:18]:18s} " + "  ".join(f"{ci[k]:5d}" for k in ks))
    print(f"  {new[:18]:18s} " + "  ".join(f"{cn[k]:5d}" for k in ks))
    moves = Counter((r[inc][1], r[new][1]) for r in rows if r[inc][1] != r[new][1])
    print("  number changes (old -> new): " + ", ".join(f"{a}->{b} x{c}" for (a, b), c in moves.most_common()))
    print("  examples of a changed pick:")
    for r in [r for r in rows if r[inc] != r[new]][:12]:
        print(f"    turn {r['turn']:2d}: {r[inc][0]} {r[inc][1]}  ->  {r[new][0]} {r[new][1]}")


if __name__ == "__main__":
    main()
