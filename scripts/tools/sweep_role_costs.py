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

**Size --max-workers by RAM, not by cores.** Each worker holds its own
similarity tensor and BOTH spymasters: measured at **1.2 GB**. The arena's
docstring says to oversubscribe past os.cpu_count() because LLM turns are
network-bound, and that is true of the cheap baselines -- but this model is
not cheap to hold, and 48 workers asks for 58 GB. On a 30 GB box that is the
OOM killer, which takes the editor down with it. Budget
(free RAM - 10 GB headroom) / 1.2 GB, and `nice` it so interactive work wins.

Usage:
    # ~2-3 h, ~$5 on gpt-oss; 10 workers = 12 GB, leaving 6 cores free
    nice -n 10 python scripts/tools/sweep_role_costs.py --n-boards 150 \\
        --guesser-pool-config configs/guesser_pool_oss120b.json \\
        --record-games cache/llm_store.db --max-workers 10

    python scripts/tools/sweep_role_costs.py --axis opponent --n-boards 20 \\
        --guesser noisy_numberbatch --max-workers 10        # free smoke test
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
BASE = {"neutral_cost": 0.2, "opponent_cost": 1.0, "assassin_cost": 10.0,
        # 0 is the incumbent: it disables the outside option entirely and
        # reproduces every result recorded before it existed.
        "outside_n": 0}

# Pinned, not inherited. Everything the spymaster reads that is NOT the axis
# under test has to be fixed here, or a change to a default silently splits the
# run into two models: `exclude_acronyms` flipped to True mid-sweep, and
# resuming would have measured the last 13 settings against a different pool
# than the first 2. Held apart from BASE because these are not swept -- they
# are the model the sweep is a statement about.
MODEL_PINS = {"exclude_acronyms": True, "k1_tiebreak": False}

# One axis at a time, so a result points at a parameter rather than at a
# corner of a grid. Values bracket the incumbent on both sides where that is
# meaningful -- a setting that loses on the low side is as informative as one
# that wins on the high side, and 0.1/5.0 are the controls that say the sweep
# is measuring the cost rather than noise.
AXES = {
    "neutral": ("neutral_cost", [0.1, 0.4, 0.7, 1.0]),
    # 0.5/0.75 matter more than the values above 1.0. The first sweep only
    # went up -- 1.5/2.0/2.5/3.0 -- on the theory that an opponent card is a
    # two-card swing and 1.0 underprices it. It measured the opposite:
    # monotonically worse as the price rose (44.7% -> 41.8%). The gradient
    # points down on every axis, and down is the side this axis never had.
    "opponent": ("opponent_cost", [0.5, 0.75, 1.5, 2.0, 2.5, 3.0]),
    # 2/7 for the same reason: ass=5 was the only setting to beat the
    # incumbent (56.9%, p=0.052), and 20/40 were the two worst results in the
    # sweep. Whatever is happening is below 10, not above it.
    "assassin": ("assassin_cost", [2.0, 5.0, 7.0, 20.0, 40.0]),
    # How many "the guesser picks something the clue never meant" alternatives
    # the reward prices, each ending the turn for zero. The decoy training
    # identifies the LEVEL of such a word (~4.1 nats below the best board
    # word) but not how many a game contains -- a real game contains none,
    # since the guesser must pick from the board -- so it is swept, not
    # derived. 1 is the control: it takes under 1% of the rate and changed
    # 1 clue in 12 offline, so an arm that moves at 1 is measuring noise.
    # REQUIRES --model-path pointing at a decoy-trained booster; against the
    # deployed model, which never saw an unrelated word in training, the
    # scores for the drawn words are extrapolation.
    "outside": ("outside_n", [1, 5, 10, 25, 50]),
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
        # Only boards played both ways count, anywhere. A board can be
        # half-present because the guesser refused to rank on one assignment,
        # and counting its surviving half toward win rate while it is absent
        # from the sign test makes the two columns disagree about which games
        # the row is describing -- with the first-move advantage landing
        # entirely on whichever side happened to survive.
        if len(gs) != 2:
            continue
        for w, killer in gs:
            wins[w] += 1
            if killer is not None:
                assassin[killer] += 1
        if gs[0][0] == gs[1][0]:
            swept[gs[0][0]] += 1
        else:
            split += 1
    paired = sum(1 for gs in boards.values() if len(gs) == 2)
    return dict(wins), dict(assassin), (dict(swept), split, paired)


def progress_path(out: Path | None, label: str) -> Path:
    """Where finished settings are remembered between runs."""
    base = out if out is not None else Path(f"{label}.json")
    return base.with_suffix(".progress.json")


def load_progress(path: Path) -> dict:
    try:
        return {r["setting"]: r for r in json.loads(path.read_text())}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}


def save_progress(path: Path, rows: list[dict]) -> None:
    """Written after every setting, not at the end -- the whole point is to
    survive a run that does not reach the end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))


def resume_state(db: Path, run: str, n_boards: int, finished: dict, name: str) -> str:
    """"complete" | "partial" | "absent" for one setting.

    Completion is read from the progress file, NOT from a row count. Counting
    rows looked obvious and was wrong: a setting that finishes with discarded
    boards writes fewer than 2*n_boards rows -- ass=5 finished with 196 and
    ass=2 with 184 of 200 -- so every setting that lost a board to the guesser
    would have been judged partial, cleared, and replayed from scratch. The
    discards are the normal case, so that bug would have made --resume delete
    exactly the work it exists to preserve.
    """
    if name in finished:
        return "complete"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        n = con.execute("select count(*) from game_records where label like ?", (run + "|%",)).fetchone()[0]
    finally:
        con.close()
    return "partial" if n else "absent"


def clear_run(db: Path, run: str) -> int:
    """Delete a partial setting's rows so the redo does not double-count.

    `GameRecordStore.add_game` keys on (spymaster_id, suite_id, seed) and this
    caller passes neither, so rows APPEND. Re-running without this leaves four
    rows per seed, `per_board` pairs none of them, and the sign test reports
    NaN instead of failing -- a silent wrong answer, which is worse than a
    crash.
    """
    con = sqlite3.connect(db)
    try:
        n = con.execute("delete from game_records where label like ?", (run + "|%",)).rowcount
        con.commit()
    finally:
        con.close()
    return n


def summarise(name: str, overrides: dict, db: Path, run: str) -> dict:
    """Rebuild a completed setting's row from the store alone, for --resume.

    Everything in the table except timing comes from the recorded games, so a
    skipped setting reports exactly what a freshly played one would.
    """
    wins, assassin, (swept, split, n_boards) = per_board(db, run)
    n_games = sum(wins.values())
    decisive = swept.get("base", 0) + swept.get(name, 0)
    lo, hi = wilson(wins.get(name, 0), n_games) if n_games else (0.0, 0.0)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        turns = [json.loads(t) for (t,) in con.execute(
            "select turns from game_records where label like ?", (run + "|%",))]
    finally:
        con.close()
    clues = [t for g in turns for t in g if t.get("team")]
    own = sum(sum(1 for w, r in t["guesses"] if r == "own") for t in clues)
    return {
        "setting": name, "overrides": overrides, "games": n_games,
        "win_rate": wins.get(name, 0) / n_games if n_games else 0.0,
        "assassin_base": assassin.get("base", 0),
        "assassin_challenger": assassin.get(name, 0),
        "own_per_clue": own / len(clues) if clues else 0.0,
        "mean_k": sum(t["number"] for t in clues) / len(clues) if clues else 0.0,
        "seconds": 0.0, "resumed": True,
        "swept_base": swept.get("base", 0), "swept_challenger": swept.get(name, 0),
        "split": split, "decisive": decisive, "boards": n_boards,
        "p_sign": binom_two_sided(swept.get(name, 0), decisive) if decisive else float("nan"),
        "ci": [lo, hi],
    }


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
    ap.add_argument("--threads-per-worker", type=int, default=1,
                    help="threads inside each worker process. 1 is the old one-game-"
                         "per-process behaviour. With an LLM guesser, e.g. "
                         "--max-workers 6 --threads-per-worker 16: processes are capped "
                         "by memory (1.13 GB each), a single process by its GIL (~2.5x a "
                         "core), and the product sets games in flight. See "
                         "codenames/two_team_arena.py::run_two_team_matchup.")
    ap.add_argument("--record-games", type=Path, default=None,
                    help="required for the paired sign test -- it needs per-board results")
    ap.add_argument("--model-path", type=Path, default=None,
                    help="booster for BOTH arms. The outside axis needs a decoy-trained "
                         "one; left unset, every arm uses the deployed model and the "
                         "outside option is scoring words it was never fitted on.")
    ap.add_argument("--label", default="role_cost_sweep")
    ap.add_argument("--resume", action="store_true",
                    help="skip settings already complete in --record-games, and clear any "
                         "partial rows for the one being redone. Without this a re-run "
                         "APPENDS (add_game keys on label alone, which never dedupes), "
                         "leaving 4 rows per seed so nothing pairs and the sign test goes NaN.")
    ap.add_argument("--out", type=Path, default=None, help="write the summary table as JSON")
    args = ap.parse_args()

    settings = []
    for axis in (args.axis or sorted(AXES)):
        key, values = AXES[axis]
        for v in values:
            settings.append({**BASE, key: v})

    seeds = list(range(args.first_seed, args.first_seed + args.n_boards))
    # Fail before the pool starts. A missing guesser raises KeyError inside the
    # ProcessPoolExecutor initializer, which surfaces only as BrokenProcessPool
    # with no cause named -- the default config carries the synthetic guessers,
    # so forgetting --guesser-pool-config looks exactly like a crash.
    from codenames.guessers.registry import load_pool
    available = load_pool(args.guesser_pool_config)
    if args.guesser not in available:
        raise SystemExit(
            f"guesser {args.guesser!r} is not in {args.guesser_pool_config}; "
            f"it has {sorted(available)}. The LLM guesser lives in "
            f"configs/guesser_pool_oss120b.json.")

    pins = dict(MODEL_PINS)
    if args.model_path is not None:
        pins["model_path"] = args.model_path
    base_spec = spymaster_spec(MODEL, **BASE, **pins)
    print("model pins: " + ", ".join(f"{k}={v}" for k, v in pins.items()), flush=True)
    print(f"{len(settings)} settings x {2*len(seeds)} games = {len(settings)*2*len(seeds)} games, "
          f"guesser {args.guesser}\n", flush=True)

    prog = progress_path(args.out, args.label)
    finished = load_progress(prog) if args.resume else {}
    if finished:
        print(f"resuming: {len(finished)} setting(s) already finished\n", flush=True)
    rows = []
    t_start = time.time()
    for i, overrides in enumerate(settings, 1):
        name = tag(overrides)
        run = f"{args.label}_{name}"
        if args.resume and args.record_games and args.record_games.exists():
            state = resume_state(args.record_games, run, len(seeds), finished, name)
            if state == "complete":
                print(f"[{i}/{len(settings)}] {name:<16} already complete -- skipping", flush=True)
                rows.append(finished.get(name) or summarise(name, overrides, args.record_games, run))
                continue
            if state == "partial":
                n = clear_run(args.record_games, run)
                print(f"[{i}/{len(settings)}] {name:<16} partial ({n} games) -- clearing and redoing",
                      flush=True)
        t0 = time.time()
        result = run_two_team_matchup(
            base_spec, spymaster_spec(MODEL, **overrides, **pins), ("base", name),
            guesser_pool_config=args.guesser_pool_config, guesser_name=args.guesser,
            seeds=seeds, max_workers=args.max_workers,
            game_record_db=args.record_games, run_label=run,
            threads_per_worker=args.threads_per_worker,
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
            "discarded": len(result.discarded_boards),
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
        save_progress(prog, rows)
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
