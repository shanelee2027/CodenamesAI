"""Sweep a spymaster's role costs by playing each setting against the incumbent.

The costs (`codenames/game.py::role_costs`) are what the spymaster believes a
wrong guess is worth, and they were never fitted -- they were inherited from
the game's own reward table, where an opponent card costs 1.0, the same as one
own word forgone. That is almost certainly too cheap: revealing an opponent
card ends the turn *and* hands them a card, a two-card swing.

**Screening, not a verdict.** Each setting plays the incumbent head to head on
the same boards, both side assignments, so the first-move advantage falls on
both equally. With ~10 settings at p<0.05 roughly one false positive is
expected, so this stage ranks candidates; the shortlist has to be re-run on
its own before anything is believed. The paired sign test on decisive boards
is the honest test here -- the Wilson interval on raw win rate ignores the
pairing and is the looser of the two.

Why the incumbent rather than a round robin: 11 settings round-robin is 55
pairings for the same number of games per pairing, and the question is "does
anything beat what we ship", not the full ordering.

Usage:
    # ~1 h, ~$5 on gpt-oss
    python scripts/tools/sweep_role_costs.py --n-boards 150 \\
        --guesser-pool-config configs/guesser_pool_oss120b.json \\
        --record-games cache/llm_store.db --max-workers 48

    python scripts/tools/sweep_role_costs.py --axis opponent --n-boards 20 \\
        --guesser noisy_numberbatch --max-workers 14        # free smoke test
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.spymasters.registry import spymaster_spec
from codenames.two_team_arena import run_two_team_matchup

# analyze_headtohead.py is a script, not a package module, so load it by path
# rather than adding __init__.py files to scripts/ just for this import.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_h2h", Path(__file__).parent / "analyze_headtohead.py")
_h2h = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_h2h)
binom_two_sided, wilson = _h2h.binom_two_sided, _h2h.wilson

MODEL = "learned_listener"

# The incumbent, from codenames/game.py's reward table.
BASE = {"neutral_cost": 0.2, "opponent_cost": 1.0, "assassin_cost": 10.0}

# One axis at a time, so a result points at a parameter rather than at a
# corner of a grid. Values bracket the incumbent on both sides where that is
# meaningful -- a setting that loses on the low side is as informative as one
# that wins on the high side, and 0.1/5.0 are the controls that say the sweep
# is measuring the cost rather than noise.
AXES = {
    "neutral": ("neutral_cost", [0.1, 0.4, 0.7, 1.0]),
    "opponent": ("opponent_cost", [1.5, 2.0, 2.5, 3.0]),
    "assassin": ("assassin_cost", [5.0, 20.0, 40.0]),
}


def tag(overrides: dict) -> str:
    """A short name for the arena and the game store. Only the changed axis
    appears, so the table reads as one column of deltas from the incumbent."""
    bits = [f"{k.split('_')[0][:3]}={v:g}" for k, v in sorted(overrides.items()) if BASE[k] != v]
    return "+".join(bits) if bits else "base-control"


def per_board(db: Path, run: str) -> tuple[dict[str, int], dict[str, int], tuple[int, int, int]]:
    """Read this run's games back out and pair them by board.

    Recomputed from the store rather than from MatchupResult because the
    paired sign test needs both side assignments of the *same* seed, which
    the aggregate counters have already summed away.
    """
    con = sqlite3.connect(db)
    rows = con.execute(
        "select label, seed, winner, turns from game_records where label like ?", (run + "|%",)
    ).fetchall()
    con.close()
    boards: dict[int, list[tuple[str, str | None]]] = defaultdict(list)
    for label, seed, winner, turns in rows:
        _, _, sides = label.partition("|")
        a, _, b = sides.partition(",")
        names = (a.split("=", 1)[1], b.split("=", 1)[1])
        won = names[0] if str(winner).upper().endswith("A") else names[1]
        killer = None
        for t in json.loads(turns):
            if t.get("ended_reason") == "assassin":
                killer = names[0] if str(t.get("team", "")).upper() == "A" else names[1]
        boards[seed].append((won, killer))

    wins: dict[str, int] = defaultdict(int)
    assassin: dict[str, int] = defaultdict(int)
    swept: dict[str, int] = defaultdict(int)
    split = 0
    for gs in boards.values():
        for w, killer in gs:
            wins[w] += 1
            if killer is not None:
                assassin[killer] += 1
        if len(gs) == 2:
            if gs[0][0] == gs[1][0]:
                swept[gs[0][0]] += 1
            else:
                split += 1
    return dict(wins), dict(assassin), (dict(swept), split, len(boards))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boards", type=int, default=150, help="boards per setting; each is played twice")
    ap.add_argument("--first-seed", type=int, default=20000,
                    help="seeds start here, clear of the training and eval-suite ranges")
    ap.add_argument("--axis", action="append", choices=sorted(AXES), default=None,
                    help="restrict to one axis (repeatable); default is all three")
    ap.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    ap.add_argument("--guesser", default="llm")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--record-games", type=Path, default=None,
                    help="required for the paired sign test -- it needs per-board results")
    ap.add_argument("--label", default="role_cost_sweep")
    ap.add_argument("--out", type=Path, default=None, help="write the summary table as JSON")
    args = ap.parse_args()

    settings = []
    for axis in (args.axis or sorted(AXES)):
        key, values = AXES[axis]
        for v in values:
            settings.append({**BASE, key: v})

    seeds = list(range(args.first_seed, args.first_seed + args.n_boards))
    base_spec = spymaster_spec(MODEL, **BASE)
    print(f"{len(settings)} settings x {2*len(seeds)} games = {len(settings)*2*len(seeds)} games, "
          f"guesser {args.guesser}\n", flush=True)

    rows = []
    t_start = time.time()
    for i, overrides in enumerate(settings, 1):
        name = tag(overrides)
        run = f"{args.label}_{name}"
        t0 = time.time()
        result = run_two_team_matchup(
            base_spec, spymaster_spec(MODEL, **overrides), ("base", name),
            guesser_pool_config=args.guesser_pool_config, guesser_name=args.guesser,
            seeds=seeds, max_workers=args.max_workers,
            game_record_db=args.record_games, run_label=run,
        )
        base_st, chal_st = result.sides["base"], result.sides[name]
        row = {
            "setting": name, "overrides": overrides,
            "games": result.n_games,
            "win_rate": chal_st.win_rate,
            "assassin_base": base_st.assassin_losses,
            "assassin_challenger": chal_st.assassin_losses,
            "own_per_clue": chal_st.correct_sum / chal_st.clues if chal_st.clues else 0.0,
            "mean_k": chal_st.clue_number_sum / chal_st.clues if chal_st.clues else 0.0,
            "seconds": time.time() - t0,
        }
        if args.record_games:
            wins, assassin, (swept, split, n_boards) = per_board(args.record_games, run)
            decisive = swept.get("base", 0) + swept.get(name, 0)
            row.update({
                "swept_base": swept.get("base", 0), "swept_challenger": swept.get(name, 0),
                "split": split, "decisive": decisive,
                "p_sign": binom_two_sided(swept.get(name, 0), decisive) if decisive else float("nan"),
            })
        lo, hi = wilson(chal_st.wins, result.n_games)
        row["ci"] = [lo, hi]
        rows.append(row)
        print(f"[{i}/{len(settings)}] {name:<16} win {chal_st.win_rate:6.1%} "
              f"[{lo:.2f},{hi:.2f}]  p={row.get('p_sign', float('nan')):.3f}  "
              f"({row['seconds']:.0f}s)", flush=True)

    rows.sort(key=lambda r: -r["win_rate"])
    print(f"\n{len(rows)} settings, {sum(r['games'] for r in rows)} games, "
          f"{(time.time()-t_start)/60:.0f} min\n")
    print(f"{'setting':<16}{'win% vs base':>13}{'95% CI':>15}{'swept':>10}{'sign p':>9}"
          f"{'assassin':>10}{'own/clue':>10}{'mean k':>8}")
    print("-" * 91)
    for r in rows:
        ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]"
        swept = f"{r.get('swept_challenger','-')}-{r.get('swept_base','-')}"
        ass = f"{r['assassin_challenger']}v{r['assassin_base']}"
        print(f"{r['setting']:<16}{r['win_rate']:>12.1%} {ci:>15}{swept:>10}"
              f"{r.get('p_sign', float('nan')):>9.3f}{ass:>10}"
              f"{r['own_per_clue']:>10.2f}{r['mean_k']:>8.2f}")
    print("\n50% is the null: each setting plays the incumbent, not the other settings.")
    print("The sign test on decisive boards is the paired test; the CI ignores the pairing.")
    print(f"With {len(rows)} comparisons, expect ~{0.05*len(rows):.1f} false positives at p<0.05 "
          "-- re-run the leaders before believing them.")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
