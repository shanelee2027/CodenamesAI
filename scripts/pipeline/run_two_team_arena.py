"""Run real two-team games (codenames/game.py::play_two_team_game) across
many seeded boards, in either of two modes.

**Self-play** -- the same spymaster+guesser pair on both sides:

    python scripts/pipeline/run_two_team_arena.py --n-boards 300 --spymaster centroid --guesser noisy_glove

**Head-to-head** -- two different spymasters, one per side:

    python scripts/pipeline/run_two_team_arena.py --n-boards 50 \\
        --spymaster centroid --vs expected_words \\
        --guesser-pool-config configs/guesser_pool_llm_sonnet.json --guesser llm \\
        --record-games cache/llm_store.db --max-workers 48

--vs plays every board twice with the sides swapped, because team A holds
9 words and moves first while team B holds 8 -- a one-sided run would
confound spymaster strength with that advantage. It reports per-spymaster
rather than pooled, since pooling two different models' clue numbers into
one average says nothing about either.

**--max-workers is not bounded by cores.** With an LLM guesser each turn
is network latency, not computation, so the useful worker count is how
many games you want in flight at once. Turns inside one game are strictly
sequential (the board changes between them), so wall-clock is roughly
(games / workers) * (turns per game) * per-call latency. Oversubscribing
well past os.cpu_count() is the intended use.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from codenames.guessers.registry import DEFAULT_POOL_CONFIG
from codenames.spymasters.registry import load_spymasters, spymaster_names
from codenames.two_team_arena import run_two_team_matchup, run_two_team_self_play

# Selected by role from configs/spymasters.json rather than by name, so a
# new entry needs no edit here. This script offers the "exploration"
# role (oracle) on top of the standard baselines; scripts/pipeline/run_arena.py
# does not -- that difference is the only reason the two lists differ.
BASE_SPYMASTER_NAMES = spymaster_names("baseline", "exploration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boards", type=int, default=20, help="number of fixed seeded boards to play (seeds 0..n-1)")
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument(
        "--guesser",
        required=True,
        help="name of one guesser in --guesser-pool-config to use on both sides, or 'mixed' -- each game "
        "independently draws a guesser uniformly from the whole pool, matching the distribution the "
        "spymaster was actually trained against (see codenames/two_team_arena.py::MIXED_GUESSER)",
    )
    parser.add_argument("--spymaster", choices=BASE_SPYMASTER_NAMES, required=True, help="the spymaster to play (team A's, with --vs)")
    parser.add_argument(
        "--vs",
        choices=BASE_SPYMASTER_NAMES,
        default=None,
        help="head-to-head against this spymaster instead of self-play. Every board is played twice "
        "with the sides swapped, so first-move advantage falls on both equally.",
    )
    parser.add_argument(
        "--param", action="append", default=[], metavar="NAME=KEY=VALUE",
        help="override one constructor param on a spymaster, e.g. --param expected_words=sigma=1.5. "
             "Values parse as float, then int, then string. The override is recorded in the default "
             "--run-label so two settings of the same model cannot collide in the game store.",
    )
    parser.add_argument("--max-turns", type=int, default=None, help="override codenames.game.DEFAULT_MAX_TURNS (per team)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="games in flight at once. NOT bounded by cores -- with an LLM guesser these are "
        "network-bound, so values well past os.cpu_count() are the point. Default: os.cpu_count().",
    )
    parser.add_argument(
        "--record-games",
        type=Path,
        default=None,
        help="persist every game's board + turn sequence to this SQLite file (codenames/llm_store.py), "
        "so a run can be inspected later without replaying it -- see scripts/tools/dump_game_records.py. "
        "Most useful when --guesser costs real money per turn (e.g. 'llm').",
    )
    parser.add_argument("--run-label", default=None, help="label stored alongside --record-games' rows (default: '<spymaster>+<guesser>')")
    args = parser.parse_args()

    if args.vs == args.spymaster:
        parser.error("--vs must name a different spymaster than --spymaster (use self-play mode instead)")

    def _coerce(v):
        for cast in (int, float):
            try:
                return cast(v) if cast is not float or "." in v or "e" in v.lower() else int(v)
            except ValueError:
                continue
        return v

    overrides: dict[str, dict] = {}
    for spec in args.param:
        name, _, rest = spec.partition("=")
        key, _, value = rest.partition("=")
        if not (name and key and value):
            parser.error(f"--param must be NAME=KEY=VALUE, got {spec!r}")
        overrides.setdefault(name, {})[key] = _coerce(value)

    entries = load_spymasters()
    for name in overrides:
        if name not in entries:
            parser.error(f"--param names unknown spymaster {name!r}")

    def spec_for(name):
        cls, kwargs = entries[name].spec
        return cls, {**kwargs, **overrides.get(name, {})}

    def label_for(name):
        ov = overrides.get(name)
        return name + ("" if not ov else "[" + ",".join(f"{k}={v}" for k, v in sorted(ov.items())) + "]")

    spymaster_cls, spymaster_kwargs = spec_for(args.spymaster)
    spymaster_label = label_for(args.spymaster)

    kwargs = {}
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    matchup_label = f"{spymaster_label}-vs-{label_for(args.vs)}" if args.vs else spymaster_label
    run_label = args.run_label if args.run_label is not None else f"{matchup_label}+{args.guesser}"
    if args.record_games is not None:
        kwargs["game_record_db"] = args.record_games
        kwargs["run_label"] = run_label

    seeds = list(range(args.n_boards))
    start = time.time()

    if args.vs is not None:
        result = run_two_team_matchup(
            spec_for(args.spymaster),
            spec_for(args.vs),
            (label_for(args.spymaster), label_for(args.vs)),
            args.guesser_pool_config,
            args.guesser,
            seeds,
            max_workers=args.max_workers,
            progress=True,
            **kwargs,
        )
        elapsed = time.time() - start
        print(
            f"\n{result.n_games} games ({result.n_boards} boards x 2 side assignments), "
            f"{label_for(args.spymaster)} vs {label_for(args.vs)}, guesser {args.guesser}, in {elapsed:.1f}s"
        )
        if result.timeouts:
            print(f"({result.timeouts} ended in timeout, counted as a win for neither)")
        print()
        header = (
            f"{'spymaster':16s} {'win%':>7s} {'as A':>7s} {'as B':>7s} {'assassin%':>10s} "
            f"{'clues':>6s} {'mean k':>7s} {'own/clue':>9s} {'own%':>7s}"
        )
        print(header)
        print("-" * len(header))
        for name in (label_for(args.spymaster), label_for(args.vs)):
            st = result.sides[name]
            as_a = st.wins_as_first / st.games_as_first if st.games_as_first else 0.0
            games_as_b = st.games - st.games_as_first
            as_b = (st.wins - st.wins_as_first) / games_as_b if games_as_b else 0.0
            print(
                f"{name:16s} {100 * st.win_rate:6.1f}% {100 * as_a:6.1f}% {100 * as_b:6.1f}% "
                f"{100 * st.assassin_rate:9.1f}% {st.clues:6d} {st.mean_clue_number:7.2f} "
                f"{st.mean_correct_per_clue:9.2f} {100 * st.own_rate:6.1f}%"
            )
        print()
        print("'as A' is win rate when moving first with 9 words; 'as B' when moving second with 8.")
        print("A large A-vs-B gap means the side advantage dominates the spymaster difference.")
        if args.record_games is not None:
            print(f"\ngames recorded to {args.record_games} under labels '{run_label}|A=...'")
            print(f"  python scripts/tools/dump_game_records.py {args.record_games} --label '{run_label}|A={label_for(args.spymaster)},B={label_for(args.vs)}'")
        return

    result = run_two_team_self_play(
        spymaster_cls,
        spymaster_kwargs,
        args.guesser_pool_config,
        args.guesser,
        seeds,
        max_workers=args.max_workers,
        **kwargs,
    )
    elapsed = time.time() - start

    print(f"{result.n_games} two-team games ({spymaster_label} + {args.guesser} on both sides) in {elapsed:.1f}s\n")
    print(f"{'assassin-hit rate':22s} {100 * result.assassin_rate:6.1f}%")
    print(f"{'half-turns (all)':22s} {result.mean_half_turns_all:6.2f}")
    turns_clean = f"{result.mean_half_turns_clean_finish:.2f}" if result.mean_half_turns_clean_finish is not None else "--"
    print(f"{'half-turns (clean finish)':22s} {turns_clean:>6s}")
    print(f"{'mean clue number':22s} {result.mean_clue_number:6.2f}")
    print(f"{'mean correct per clue':22s} {result.mean_correct_per_clue:6.2f}")
    print()
    print("per-guess role breakdown (of every word actually guessed, pooled across both teams):")
    print(
        f"  own {100 * result.guess_own_rate:5.1f}%   opponent {100 * result.guess_opponent_rate:5.1f}%   "
        f"neutral {100 * result.guess_neutral_rate:5.1f}%   assassin {100 * result.guess_assassin_rate:5.1f}%"
    )


if __name__ == "__main__":
    main()
