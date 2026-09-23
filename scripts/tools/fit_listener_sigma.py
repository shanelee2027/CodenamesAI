"""What `sigma` best describes a listener whose rankings we already paid for?

Reads cached rankings out of cache/llm_store.db and fits the one parameter of
`expected_words`'s guesser model to them (codenames/listener_fit.py). Costs
nothing: every response was bought by an earlier run.

    python scripts/tools/fit_listener_sigma.py
    python scripts/tools/fit_listener_sigma.py --model claude-sonnet-5 --curve

Read codenames/listener_fit.py's docstring before quoting a number from this.
The fit is descriptive -- it says how far a listener's choices sit from
numberbatch's z-order -- and that is NOT automatically the sigma a spymaster
should play with.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats
from codenames.listener_fit import (
    calibration,
    fit_sigma,
    likelihood_interval,
    load_observations,
    profile,
)
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

DB = PROJECT_ROOT / "cache" / "llm_store.db"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--model", action="append", default=None,
                    help="repeatable; default: every model in the store with enough rows")
    ap.add_argument("--space", default="numberbatch", help="must match the spymaster's space")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--include-numberless", action="store_true",
                    help="also use rows from score_candidates (prompt omits the count, so a "
                         "different question -- off by default)")
    ap.add_argument("--curve", action="store_true", help="print the log-likelihood profile")
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)

    if args.model:
        models = args.model
    else:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        models = [m for m, n in conn.execute(
            "SELECT model, COUNT(*) FROM responses GROUP BY model ORDER BY 2 DESC") if n >= args.min_rows]
        conn.close()

    print(f"space={args.space}  db={args.db}")
    print(f"{'listener':34s} {'n':>6s} {'sigma':>7s} {'95% LI':>16s} "
          f"{'obs top1':>9s} {'sim top1':>9s} {'obs rank':>9s} {'sim rank':>9s}")
    print("-" * 110)

    for model in models:
        obs = load_observations(args.db, model, sims, stats, space=args.space,
                                require_number=not args.include_numberless)
        if len(obs) < args.min_rows:
            print(f"{model:34s} {len(obs):6d}   (too few usable rows)")
            continue
        best, _ = fit_sigma(obs)
        lo, hi = likelihood_interval(obs, best)
        cal = calibration(obs, best)
        print(f"{model:34s} {len(obs):6d} {best:7.3f} {f'[{lo:.2f}, {hi:.2f}]':>16s} "
              f"{cal['observed_top1_rate']:9.3f} {cal['simulated_top1_rate']:9.3f} "
              f"{cal['observed_mean_zrank']:9.2f} {cal['simulated_mean_zrank']:9.2f}")

        if args.curve:
            grid = np.geomspace(max(best / 4, 0.05), best * 4, 25)
            ll = profile(obs, grid)
            peak = ll.max()
            print(f"    log-likelihood profile (peak {peak:.1f}):")
            for s, v in zip(grid, ll):
                bar = "#" * int(max(0, 40 + (v - peak) / max(abs(peak) * 0.002, 1e-9)))
                print(f"      sigma={s:6.3f}  dLL={v - peak:8.2f}  {bar}")

    print()
    print("'obs top1'  how often the listener's pick was numberbatch's own top word.")
    print("'sim top1'  what the fitted sigma predicts for that rate.")
    print("'obs/sim rank' mean z-rank of the chosen word (0 = numberbatch's favourite).")
    print("A gap between obs and sim means the Gaussian-noise model does not fit, and")
    print("the sigma is absorbing systematic disagreement rather than measuring noise.")
    print("See codenames/listener_fit.py on why that matters before quoting this number.")


if __name__ == "__main__":
    main()
