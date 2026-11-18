"""Policy iteration for the GBT spymaster (docs/versions/gbt_win_value.md).

The same loop as scripts/pipeline/gaussian_policy_iteration.py, with the GBT
in place of the Gaussian model:

Round 0: `fresh_split_listener` (expected reward) plays itself; V_0(a, b) is
fitted to who won from each score (a, b).
Round t: `gbt_win_value` with V_{t-1} plays itself; V_t from its games.
Stops when V changes by less than `--tol` (game-weighted mean absolute change
over cells) or after `--rounds`.

- **Guesser.** Every game uses `sim:fresh_split`, a simulated guesser that
  samples each pick from the GBT's scores with its per-pick temperature
  (codenames/guessers/listener_sample.py), so in these games the spymaster's
  model of the guesser is exactly true, as the Gaussian guesser was for the
  Gaussian spymaster. The games cost no API calls.
- **Boards.** Dealt from the 150 tuning words (`--vocab split_tune`), seeds
  0..n-1, both seatings, the same boards every round. Not the training words,
  on which the GBT is sharper than on new words (its temperatures were fitted
  on the tuning words), and not the test words, which stay for evaluation.
- **Shortlist.** 1000 clues from the Gaussian first stage, not the whole pool:
  a whole-pool turn takes ~10 s against ~0.75 s, and at 1000 the pick equals
  the whole-pool pick on 95% of turns (docs/log.md, 2026-11-16).
- **V fit.** Logistic by default (`build_win_value.py --logistic`): in the
  Gaussian runs the isotonic and shrunk tables followed each round's noise and
  flipped near-tied choices of the number, and only the logistic V gave smooth
  number choices (docs/log.md, 2026-11-14/15). `--fit isotonic` or `shrink`
  are there for comparison.

Games go to `--db` (default cache/selfplay_games.db), never cache/llm_store.db.

    python scripts/pipeline/gbt_policy_iteration.py --boards 1000 --rounds 4 --workers 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaussian_policy_iteration import build_v  # noqa: E402

OUT_DIR = ROOT / "cache" / "policy_iteration" / "gbt"
GUESSER = "sim:fresh_split"


def play(name: str, extra: list[str], label: str, args) -> None:
    # The arena takes the same spymaster on both sides only with side-scoped params.
    cmd = [sys.executable, str(ROOT / "scripts" / "pipeline" / "run_two_team_arena.py"),
           "--n-boards", str(args.boards), "--spymaster", name, "--vs", name,
           "--param-a", f"shortlist={args.shortlist}", "--param-b", f"shortlist={args.shortlist}",
           "--param", f"{name}=gpu_trees=True", *extra,
           "--vocab", args.vocab, "--guesser", GUESSER, "--max-workers", str(args.workers),
           "--record-games", str(args.db), "--run-label", label]
    t = time.time()
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if "games (" in l or "discarded" in l or "#A " in l or "#B " in l]
    print(f"  {label}: [{time.time() - t:.0f}s]\n    " + ("\n    ".join(lines) or out.stdout[-500:] + out.stderr[-2000:]),
          flush=True)
    if out.returncode != 0 or not any("games (" in l for l in lines):
        raise SystemExit(f"arena failed for {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=1000, help="boards per round, each in both seatings")
    ap.add_argument("--rounds", type=int, default=4, help="most W rounds after round 0")
    ap.add_argument("--tol", type=float, default=0.005, help="stop when V's game-weighted mean change is below this")
    ap.add_argument("--workers", type=int, default=6, help="~3.4 GB of RAM each at peak")
    ap.add_argument("--shortlist", type=int, default=1000)
    ap.add_argument("--vocab", default="split_tune")
    ap.add_argument("--db", type=Path, default=ROOT / "cache" / "selfplay_games.db")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where V_round{t}.npz go")
    ap.add_argument("--fit", choices=["isotonic", "logistic", "shrink"], default="logistic")
    ap.add_argument("--prior", type=float, default=300.0, help="games' weight of the logistic prior for --fit shrink")
    ap.add_argument("--reuse-round0", default=None, help="label of round-0 games already in --db, not replayed")
    ap.add_argument("--start-round", type=int, default=1,
                    help="resume: first W round to play; V_round{start-1}.npz must exist in --out-dir")
    ap.add_argument("--run", default=time.strftime("%Y%m%d-%H%M"), help="tag in every game label of this run")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_round == 1:
        label = args.reuse_round0 or f"gbtpi-{args.run}-r0"
        print(f"round 0: fresh_split_listener self-play, {args.boards} boards ({args.vocab}), both seatings", flush=True)
        if not args.reuse_round0:
            play("fresh_split_listener", [], label, args)
        prev = build_v(label, out_dir / "V_round0.npz", args)
    else:
        prev = np.load(out_dir / f"V_round{args.start_round - 1}.npz")

    for t in range(args.start_round, args.rounds + 1):
        label = f"gbtpi-{args.run}-r{t}"
        value = out_dir / f"V_round{t - 1}.npz"
        print(f"round {t}: gbt_win_value with {value.name} self-play", flush=True)
        play("gbt_win_value", ["--param", f"gbt_win_value=value_path={value}"], label, args)
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
