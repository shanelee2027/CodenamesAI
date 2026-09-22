"""Read the blind one-clue study and say whether the arms differ.

`scripts/tools/play_server.py /eval` appends one row per guessing turn to
`cache/human_eval.jsonl`. This compares the arms on four quantities, the first
being the one the study is built around:

    reward      realised reward of the turn under the game's own yardstick
                (codenames/game.py::ROLE_REWARD): +1 per own word found, plus
                the ending card's value -- -0.2 bystander, -1 red, -10 assassin,
                0 for stopping, exhausting the guesses or clearing the board.
                This is what the spymaster maximises, measured on a human.
    own         own words found per clue.
    first own   the first pick landed on an own word (turns with a pick).
    stopped     the guesser stopped by choice -- the behaviour the outside
                option models and gpt-oss never shows.

Differences come with a bootstrap 95% interval and a two-sided permutation
p-value over turns. The script then states how many positions per arm the
observed spread and gap would need for 80% power -- the pilot's real output.

Usage:
    python scripts/tools/analyze_human_eval.py
    python scripts/tools/analyze_human_eval.py --player alice --since 2026-09-23
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

# Runs from the repo (scripts/tools/) or from a demo bundle (top level), as
# play_server.py does -- pick whichever candidate holds the package.
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next((c for c in (_HERE.parents[1], _HERE) if (c / "codenames").is_dir()), _HERE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Role
from codenames.game import ROLE_REWARD
from codenames.similarity import DEFAULT_CACHE_DIR

END_VALUE = {"opponent": ROLE_REWARD[Role.OPPONENT], "neutral": ROLE_REWARD[Role.NEUTRAL],
             "assassin": ROLE_REWARD[Role.ASSASSIN]}


def reward(row: dict) -> float:
    return row["own_found"] * ROLE_REWARD[Role.OWN] + END_VALUE.get(row["ended_by"], 0.0)


METRICS = {
    "reward": (lambda r: reward(r), lambda r: True),
    "own": (lambda r: float(r["own_found"]), lambda r: True),
    "first own": (lambda r: float(r["first_pick_own"]), lambda r: r["first_pick_own"] is not None),
    "stopped": (lambda r: float(r["stopped"]), lambda r: True),
}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def boot_diff(a: list[float], b: list[float], reps: int, rng: random.Random) -> tuple[float, float]:
    ds = sorted(mean([rng.choice(b) for _ in b]) - mean([rng.choice(a) for _ in a])
                for _ in range(reps))
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)]


def perm_p(a: list[float], b: list[float], reps: int, rng: random.Random) -> float:
    obs = abs(mean(b) - mean(a))
    pool, na = a + b, len(a)
    hits = 0
    for _ in range(reps):
        rng.shuffle(pool)
        hits += abs(mean(pool[na:]) - mean(pool[:na])) >= obs - 1e-12
    return (hits + 1) / (reps + 1)


def n_for_power(delta: float, s: float) -> float:
    """Positions per arm for 80% power at alpha 0.05, two-sided: 2(z_a + z_b)^2 s^2 / d^2."""
    if not delta or not s or math.isnan(s):
        return float("inf")
    return 2 * (1.96 + 0.8416) ** 2 * s ** 2 / delta ** 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_CACHE_DIR / "human_eval.jsonl")
    ap.add_argument("--player", default=None, help="only this player's turns")
    ap.add_argument("--since", default=None, help="ISO date; only turns on or after it")
    ap.add_argument("--reps", type=int, default=4000)
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"no study data at {args.log} yet -- play some turns at /eval")
    rows = [json.loads(ln) for ln in args.log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.player:
        rows = [r for r in rows if r.get("player") == args.player]
    if args.since:
        rows = [r for r in rows if r["ts"] >= args.since]
    arms = sorted({r["arm"] for r in rows})
    print(f"{len(rows)} turns   players: {dict(Counter(r.get('player') or '(unnamed)' for r in rows))}")
    if len(arms) != 2:
        raise SystemExit(f"need exactly two arms to compare, found {arms}")
    a_name, b_name = arms
    A = [r for r in rows if r["arm"] == a_name]
    B = [r for r in rows if r["arm"] == b_name]
    print(f"arms: {a_name} (n={len(A)})  vs  {b_name} (n={len(B)})\n")

    print(f"{'':12s} {a_name:>12s} {b_name:>12s} {'diff (B-A)':>11s}   {'95% CI':>16s}   perm p")
    print("-" * 78)
    rng = random.Random(0)
    power = {}
    for name, (f, keep) in METRICS.items():
        xa = [f(r) for r in A if keep(r)]
        xb = [f(r) for r in B if keep(r)]
        d = mean(xb) - mean(xa)
        lo, hi = boot_diff(xa, xb, args.reps, rng) if xa and xb else (float("nan"),) * 2
        p = perm_p(xa, xb, args.reps, rng) if xa and xb else float("nan")
        print(f"{name:12s} {mean(xa):12.3f} {mean(xb):12.3f} {d:+11.3f}   [{lo:+.3f},{hi:+.3f}]   {p:.3f}")
        power[name] = (d, sd(xa + xb))

    print(f"\n{'':12s} {'mean k':>12s}")
    for nm, grp in ((a_name, A), (b_name, B)):
        ends = Counter(r["ended_by"] for r in grp)
        print(f"{nm:12s} {mean([r['number'] for r in grp]):12.2f}   ended: "
              + ", ".join(f"{k} {v}" for k, v in ends.most_common()))

    print("\nPositions per arm for 80% power, if the observed gap is the true one:")
    for name, (d, s) in power.items():
        n = n_for_power(d, s)
        have = min(len(A), len(B))
        tail = "already there" if n <= have else f"~{n - have:.0f} more per arm" if n < 1e6 else "no gap to detect"
        print(f"  {name:12s} gap {d:+.3f}, sd {s:.2f} -> {n:7.0f}   ({tail})")
    print("\nA pilot's gap is itself noisy: treat these as the order of magnitude, and\n"
          "decide the full sample size BEFORE looking at more results, not after.")


if __name__ == "__main__":
    main()
