"""Does the decoy signal survive changing how many decoys there are?

The decoy anchor rests on Luce/IIA: the level a clue sits at is a property of
the clue, so mixing in more decoys must not change it. That assumption cannot
be checked by comparing raw decoy-first rates across D, because the rate is
mechanically higher when there are more decoys to hit -- and normalising by
chance does not fix it either, since more draws mean more chances one of them
happens to relate to the clue and outrank the word the clue meant. Subsampling
the 15-decoy probe showed exactly that: observed/chance drifts 0.43 -> 0.52
from D=1 to D=15 under pure IIA subsampling, so drift alone proves nothing.

The test that does work is internal. Take the positions COLLECTED at D=10 and
subsample their decoys down to 2 and 5. Under IIA the induced sub-ranking is
what the teacher would have produced with that many decoys, so the subsampled
figure must match the figure from positions ACTUALLY collected at 2 and 5.
Same clue distribution, same prompt, same pipeline -- the only difference is
whether the decoys were absent when the teacher answered, or removed
afterwards. A gap between them is IIA failing, which is the thing that would
invalidate reading the level off decoys at all.

Board size varies (positions are sampled across the arc of a game), so the
chance rate D/(D+B) differs per row and raw proportions are not comparable.
Everything below is therefore observed/expected: total decoy-first events over
the sum of per-row chance rates. O/E = 1 is no information.

Usage:
    python scripts/tools/analyze_decoy_invariance.py cache/training_data/decoys.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def oe(rows: list[tuple[int, float]]) -> float:
    """Observed over expected, for (was_decoy_first, chance_rate) pairs."""
    exp = sum(e for _, e in rows)
    return (sum(o for o, _ in rows) / exp) if exp else float("nan")


def boot(rows: list[tuple[int, float]], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = sorted(oe([rows[rng.randrange(len(rows))] for _ in range(len(rows))])
                  for _ in range(reps))
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


def first_pick(row: dict, keep: set[str]) -> tuple[int, float] | None:
    """`(decoy_won, chance)` for this row with only `keep` decoys present.

    Decoys not kept are deleted from the ranking, which under IIA is what the
    teacher would have produced had they never been shown.
    """
    dropped = set(row["decoys"]) - keep
    rank = [w for w in row["ranking"] if w not in dropped]
    if not rank:
        return None
    B = row["n_board"]
    return int(rank[0] in keep), len(keep) / (len(keep) + B)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--reps", type=int, default=200, help="subsample draws per D")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ranking")]
    by_d: dict[int, list[dict]] = {}
    for r in rows:
        by_d.setdefault(r["n_decoys"], []).append(r)

    print(f"{len(rows)} positions   " + "  ".join(f"D={d}:{len(v)}" for d, v in sorted(by_d.items())))
    print("\nO/E = decoy-first events over chance. 1.00 = no information.\n")
    print("  D    source        n     O/E    95% CI")
    print("  " + "-" * 44)

    collected = {}
    for d, group in sorted(by_d.items()):
        pts = [p for p in (first_pick(r, set(r["decoys"])) for r in group) if p]
        collected[d] = oe(pts)
        lo, hi = boot(pts)
        print(f" {d:2d}    collected  {len(pts):4d}  {collected[d]:6.2f}   [{lo:.2f},{hi:.2f}]")

    big = max(by_d)
    print()
    for d in sorted(by_d):
        if d >= big:
            continue
        rng = random.Random(1234)
        vals = []
        for _ in range(args.reps):
            pts = [p for p in (first_pick(r, set(rng.sample(r["decoys"], d)))
                               for r in by_d[big]) if p]
            vals.append(oe(pts))
        est = sum(vals) / len(vals)
        gap = est - collected[d]
        print(f" {d:2d}    from D={big}  {len(by_d[big]):4d}  {est:6.2f}   "
              f"gap vs collected {gap:+.2f}")

    print("\nA gap near zero means IIA holds and the level can be read off any D.")
    print("A systematic gap means decoys interact, and the level is not a clue property.")


if __name__ == "__main__":
    main()
