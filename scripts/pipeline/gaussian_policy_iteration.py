"""Policy iteration for the Gaussian spymaster (thesis Section 4).

Round 0: `expected_words` at the fitted sigma plays itself; V_0(a, b) is the
fraction of positions with score (a, b) from which the side to move won.
Round t: `gaussian_win_value` with V_{t-1} plays itself; V_t from its games.
Stops when V changes by less than `--tol` (game-weighted mean absolute
change over cells) or after `--rounds`.

Every game uses the simulated guesser that follows the Gaussian model
exactly at the fitted sigma 2.02 (configs/guesser_gaussian.json), so the games
cost nothing. `--sigma` sets the spymaster's sigma only (default 2.02, the
same); the guesser stays at 2.02, the best model of the real guesser, so a
spymaster at another sigma gets the V of its own play against that guesser.
Write such a run to its own `--out-dir`. They are recorded in their own database (`--db`, default
cache/selfplay_games.db), not cache/llm_store.db, which holds the paid
results. V is pure weighted isotonic regression
(scripts/data/build_win_value.py --isotonic). Every round plays the same
boards (seeds 0..n-1 of the 800 online words, both seatings), so the rounds
differ only in the policy.

    python scripts/pipeline/gaussian_policy_iteration.py --boards 2500 --rounds 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cache" / "policy_iteration"
SIGMA = 2.02
GUESSER = "gaussian_numberbatch_s2.02"
# Sigma goes to each side separately: the arena takes the same spymaster on
# both sides only with side-scoped params (its "null control").
COMMON = ["--param", "{name}=clue_stats_stem=clue_stats_online", "--param", "{name}=max_number=None",
          "--param", "{name}=prefilter=1000", "--param-a", "sigma={sigma}", "--param-b", "sigma={sigma}"]


def play(name: str, extra: list[str], label: str, args) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "pipeline" / "run_two_team_arena.py"),
           "--n-boards", str(args.boards), "--spymaster", name, "--vs", name,
           *[c.format(name=name, sigma=args.sigma) for c in COMMON], *extra,
           "--vocab", "online", "--guesser-pool-config", str(ROOT / "configs" / "guesser_gaussian.json"),
           "--guesser", GUESSER, "--max-workers", str(args.workers),
           "--record-games", str(args.db), "--run-label", label]
    t = time.time()
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if "games (" in l or "discarded" in l]
    print(f"  {label}: {' '.join(lines) or out.stdout[-500:] + out.stderr[-2000:]} [{time.time() - t:.0f}s]", flush=True)
    if out.returncode != 0 or not any("games (" in l for l in lines):
        raise SystemExit(f"arena failed for {label}")


def build_v(label: str, out: Path, args) -> np.ndarray:
    cmd = [sys.executable, str(ROOT / "scripts" / "data" / "build_win_value.py"), "--db", str(args.db),
           "--label", label, "--no-suite", "--isotonic", "--out", str(out)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    print("\n".join("    " + l for l in res.stdout.splitlines()), flush=True)
    return np.load(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=2500, help="boards per round, each in both seatings")
    ap.add_argument("--rounds", type=int, default=4, help="most W rounds after round 0")
    ap.add_argument("--tol", type=float, default=0.01, help="stop when V's game-weighted mean change is below this")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--db", type=Path, default=ROOT / "cache" / "selfplay_games.db")
    ap.add_argument("--sigma", type=float, default=SIGMA, help="the spymaster's sigma; the guesser stays at 2.02")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where V_round{t}.npz go")
    ap.add_argument("--run", default=time.strftime("%Y%m%d-%H%M"), help="tag in every game label of this run")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f"pi-{args.run}-r0"
    print(f"round 0: expected_words (sigma {args.sigma}) self-play, {args.boards} boards", flush=True)
    play("expected_words", [], label, args)
    prev = build_v(label, out_dir / "V_round0.npz", args)

    for t in range(1, args.rounds + 1):
        label = f"pi-{args.run}-r{t}"
        value = out_dir / f"V_round{t - 1}.npz"
        print(f"round {t}: gaussian_win_value with {value.name} self-play", flush=True)
        play("gaussian_win_value", ["--param", f"gaussian_win_value=value_path={value}"], label, args)
        cur = build_v(label, out_dir / f"V_round{t}.npz", args)
        inner = (slice(1, None), slice(1, None))
        diff = np.abs(cur["V"][inner] - prev["V"][inner])
        weight = cur["n"][inner]
        mean = float((diff * weight).sum() / weight.sum())
        print(f"  change from V_round{t - 1}: game-weighted mean {mean:.4f}, max {diff.max():.3f}", flush=True)
        prev = cur
        if mean < args.tol:
            print(f"converged after round {t}", flush=True)
            break


if __name__ == "__main__":
    main()
