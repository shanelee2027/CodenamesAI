"""Compare policies on game files from train_win_actor_critic.py `play`.

    python scripts/tools/analyze_win_games.py cache/training_data/win_games_val_init.jsonl \\
        cache/training_data/win_games_val_trained.jsonl

For each file: the agent's win rate against the learned listener (Wilson
95% interval), how its games ended, and its turns (number announced, own
words found, how each turn ended). Validation files share their seeds and
seats, so for two files the comparison is paired: the sign test is on the
seeds exactly one of the two won.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from codenames.stats import binom_two_sided, mean, wilson

MISSES = ("neutral", "opponent", "assassin")


def load(path: Path) -> dict[int, dict]:
    with open(path) as f:
        games = [json.loads(line) for line in f if line.strip()]
    return {g["seed"]: g for g in games if not g.get("error") and g.get("winner")}


def summary(games: dict[int, dict]) -> dict:
    won = [g["winner"] == g["agent"] for g in games.values()]
    ends = Counter()
    turns = []
    for g in games.values():
        last = g["turns"][-1]
        if last["ended"] == "assassin":
            ends["agent hit assassin" if last["mover"] == g["agent"] else "opponent hit assassin"] += 1
        else:
            ends["agent cleared" if g["winner"] == g["agent"] else "opponent cleared"] += 1
        turns += [t for t in g["turns"] if t["agent"]]
    lo, hi = wilson(sum(won), len(won))
    return {"n": len(won), "win": mean(won), "ci": (lo, hi), "ends": dict(ends),
            "agent_turns_per_game": len(turns) / max(1, len(games)),
            "mean_k": mean([t["number"] for t in turns]),
            "k_over_4": mean([t["number"] > 4 for t in turns]),
            # A turn's guesses are own words until the one that ended it on a miss.
            "own_per_turn": mean([len(t["guesses"]) - (t["ended"] in MISSES) for t in turns]),
            "turn_ended": dict(Counter(t["ended"] for t in turns))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    runs = {p: load(p) for p in args.files}
    for p, games in runs.items():
        s = summary(games)
        print(f"{p.name}: {s['n']} games, won {s['win']:.1%} [{s['ci'][0]:.1%}, {s['ci'][1]:.1%}]")
        print(f"   endings {s['ends']}")
        print(f"   agent turns/game {s['agent_turns_per_game']:.2f}, mean k {s['mean_k']:.2f} "
              f"(k > 4 on {s['k_over_4']:.1%}), own words/turn {s['own_per_turn']:.2f}")
        print(f"   agent turns ended {s['turn_ended']}")
    if len(runs) == 2:
        (pa, a), (pb, b) = runs.items()
        common = sorted(set(a) & set(b))
        only_a = sum(a[s]["winner"] == a[s]["agent"] and b[s]["winner"] != b[s]["agent"] for s in common)
        only_b = sum(b[s]["winner"] == b[s]["agent"] and a[s]["winner"] != a[s]["agent"] for s in common)
        print(f"\npaired on {len(common)} seeds: only {pa.name} won {only_a}, only {pb.name} won {only_b}; "
              f"sign test p = {binom_two_sided(only_b, only_a + only_b):.3f}")


if __name__ == "__main__":
    main()
