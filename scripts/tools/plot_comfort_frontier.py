"""The comfort arena's scoreboard: win rate against comfort, one point per
model (docs/log.md, "The comfort arena"). No API calls.

**Win rate.** Each model's share of its 200 games against
assoc_profile_listener on the suite (default: the comfort arena,
configs/eval_suite_gptoss_subset_guess.json). The 95% interval is a
bootstrap over boards, since a board's two games are paired.

**Comfort.** For every clue the model gave, the two-step guesser's first
answer is already stored (codenames/relatedness.py, the words it says the
clue clearly points at, asked without the number). Comfort is the share of
the number those words cover, min(|clear set|, k) / k, averaged over the
model's turns: 1 when the clear words always cover the number. It is the
same quantity as the probe's sure ÷ number, from the question the arena's
guesser actually acted on. Taken from the stored answers rather than from
how the turn ended, because a turn cut short by a wrong guess never shows
whether the clear set would have run out. Interval: bootstrap over boards.

The incumbent's point pools its games across all pairings (its win rate is
50% by construction). The Pareto frontier joins the models no other model
beats on both axes. If a comfort probe report for the same suite exists
(scripts/tools/probe_comfort.py --suite), its sure ÷ number is printed
beside the stored-answer comfort as a check.

    python scripts/tools/plot_comfort_frontier.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "cache" / "llm_store.db"
DEFAULT_SUITE = PROJECT_ROOT / "configs" / "eval_suite_gptoss_subset_guess.json"
GUESSER_ID = "deepinfra/openai/gpt-oss-120b+effort=low"     # the relatedness answers' identity
INCUMBENT = "assoc_profile_listener:17861186a927"
NAMES = {
    INCUMBENT: "assoc_profile (incumbent)",
    "win_prob_listener:105dc03b7f48": "win_prob (assoc)",
    "pick_index_lookahead_listener:1cd80f3e8da1": "pick_index_lookahead",
    "profile_temperature_listener:dcc7d23c9715": "profile_temperature",
    "stop_listener:67e21b3399da": "stop, strict",
    "stop_listener:ee718b5d187a": "stop",
    "stop_net_words_listener:67e21b3399da": "stop_net, strict",
    "stop_net_words_listener:ee718b5d187a": "stop_net",
    "overpromise_listener:99e8202fc302": "overpromise 0.5, strict",
    "overpromise_listener:d4572b2f5e53": "overpromise 0.5",
    "overpromise_listener:5634934c5eae": "overpromise 1.0, strict",
    "win_prob_overpromise_listener:d629472b76bf": "win_prob + overpromise 0.5, strict",
    "win_prob_overpromise_listener:4a004e7828b3": "win_prob + overpromise 1.0, strict",
    "margin_value_overpromise_listener:bd4c9454e474": "margin V + overpromise 0.5, strict",
    "margin_value_overpromise_listener:0288c5698292": "margin V + overpromise 1.0, strict",
    "floored_overpromise_listener:646834c252ac": "floored margin V + overpromise 1.0, strict",
}


def boot(per_board: dict[int, list[float]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over all items, and a 95% bootstrap interval resampling boards."""
    boards = list(per_board.values())
    flat = np.concatenate([np.asarray(b, float) for b in boards])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(boards), len(boards))
        means.append(np.concatenate([np.asarray(boards[i], float) for i in pick]).mean())
    return float(flat.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Names of the points no other point beats on both comfort and win rate."""
    return [n for n, x, y in points
            if not any(x2 >= x and y2 >= y and (x2 > x or y2 > y) for _, x2, y2 in points)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = ap.parse_args()

    from codenames.board import Board, load_holdout_wordlist
    from codenames.eval_suite import load_eval_suite
    from codenames.relatedness import RelatednessStore

    suite = load_eval_suite(args.suite)
    store = RelatednessStore(DB)
    vocab = load_holdout_wordlist()
    con = sqlite3.connect(DB, timeout=60)

    wins: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    comfort: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing = repeats = 0
    seen: set[tuple] = set()
    words_of: dict[int, list[str]] = {}
    for sid, seed, tj, winner in con.execute(
            "SELECT spymaster_id, seed, turns, winner FROM game_records WHERE suite_id = ?", (suite.suite_id,)):
        seat = dict(kv.split("=", 1) for kv in sid.split(","))
        pair = tuple(sorted(seat.values()))
        for team in "AB":
            wins[seat[team]][(pair, seed)].append(float(winner == team))
        if seed not in words_of:
            words_of[seed] = Board.generate(seed=seed, vocabulary=vocab).words
        revealed: set[str] = set()
        for t in json.loads(tj):
            left = [w for w in words_of[seed] if w not in revealed]
            # A pass leaves the board as it was, and neither spymaster changes
            # its clue after one, so a position can repeat until the turn cap
            # (one game of seed 55 ran 80 turns of "cereal" 4 / "wrong" 3).
            # Each position counts once.
            pos = (sid, seed, t["team"], frozenset(revealed), t["clue"], int(t["number"]))
            if pos in seen:
                repeats += 1
                continue
            seen.add(pos)
            got = store.get(GUESSER_ID, t["clue"], left)
            if got is None:
                missing += 1
            else:
                k = int(t["number"])
                comfort[seat[t["team"]]][seed].append(min(len(got[0]), k) / k)
            revealed |= {w for w, _ in t["guesses"]}

    sure = {}
    probe = PROJECT_ROOT / "cache" / "report" / f"comfort_probe_{suite.name}_100_sure.json"
    if probe.exists():
        by = defaultdict(list)
        for t in json.loads(probe.read_text())["turns"]:
            if (t.get("answers") or {}).get("sure"):
                by[t["spymaster"]].append(min(len(t["answers"]["sure"]["sure"]), t["k"]) / t["k"])
        sure = {m: float(np.mean(v)) for m, v in by.items()}

    # Only boards played both ways, as the suite's own summary counts them: a
    # board refused after one game leaves a lone game behind.
    wins = {m: {key: v for key, v in b.items() if len(v) == 2} for m, b in wins.items()}
    rows = []
    for m in wins:
        w, wlo, whi = boot(wins[m])
        c, clo, chi = boot(comfort[m])
        rows.append({"id": m, "name": NAMES.get(m, m), "win": w, "win_ci": [wlo, whi],
                     "comfort": c, "comfort_ci": [clo, chi], "games": sum(map(len, wins[m].values())),
                     "turns": sum(map(len, comfort[m].values())), "sure_probe": sure.get(m)})
    rows.sort(key=lambda r: -r["win"])
    front = set(frontier([(r["name"], r["comfort"], r["win"]) for r in rows]))

    print(f"suite {suite.name} ({suite.suite_id}); {missing} turns without a stored first answer; "
          f"{repeats} repeated positions counted once\n")
    print(f"{'model':28s} {'win%':>6s} {'95% CI':>15s} {'comfort':>8s} {'95% CI':>15s} {'sure/k probe':>12s}  frontier")
    for r in rows:
        s = f"{r['sure_probe']:.3f}" if r["sure_probe"] is not None else "-"
        print(f"{r['name']:28s} {r['win']:6.1%} [{r['win_ci'][0]:.3f}, {r['win_ci'][1]:.3f}] {r['comfort']:8.3f} "
              f"[{r['comfort_ci'][0]:.3f}, {r['comfort_ci'][1]:.3f}] {s:>12s}  {'*' if r['name'] in front else ''}")
    if sure:
        both = [r for r in rows if r["sure_probe"] is not None]
        rho = np.corrcoef(np.argsort(np.argsort([r["comfort"] for r in both])),
                          np.argsort(np.argsort([r["sure_probe"] for r in both])))[0, 1]
        print(f"\nstored-answer comfort vs probe sure/k across {len(both)} models: Spearman {rho:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for r in rows:
        inc = r["id"] == INCUMBENT
        color = "black" if inc else ("tab:red" if r["name"] in front else "tab:blue")
        ax.errorbar(r["comfort"], 100 * r["win"],
                    xerr=[[r["comfort"] - r["comfort_ci"][0]], [r["comfort_ci"][1] - r["comfort"]]],
                    yerr=[[100 * (r["win"] - r["win_ci"][0])], [100 * (r["win_ci"][1] - r["win"])]],
                    fmt="s" if inc else "o", color=color, ecolor=color, alpha=0.8, capsize=2, ms=7)
        ax.annotate(r["name"], (r["comfort"], 100 * r["win"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    fr = sorted((r for r in rows if r["name"] in front), key=lambda r: r["comfort"])
    ax.plot([r["comfort"] for r in fr], [100 * r["win"] for r in fr], "--", color="tab:red", lw=1, label="frontier")
    ax.axhline(50, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("comfort: share of the number covered by the words gpt-oss calls clear (higher = clearer)")
    ax.set_ylabel("win % against assoc_profile_listener")
    ax.set_title(f"Win rate vs comfort, two-step guess-only guesser ({suite.name})", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = PROJECT_ROOT / "cache" / "report" / f"comfort_frontier_{suite.name}"
    fig.savefig(out.with_suffix(".png"), dpi=130)
    out.with_suffix(".json").write_text(json.dumps({"suite": suite.name, "rows": rows,
                                                    "frontier": sorted(front)}, indent=1))
    print(f"\nsaved -> {out.with_suffix('.png')} and .json")


if __name__ == "__main__":
    main()
