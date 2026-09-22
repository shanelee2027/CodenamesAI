"""Pack a self-contained folder that plays the real model on any laptop.

The point is a demo that survives being moved: `cache/` is gitignored, so a
fresh clone has the code and none of the artifacts, and rebuilding them means
re-downloading several embedding sets. This copies exactly what the play
server reads and nothing else -- notably NOT `cache/llm_store.db`, which is
78 MB of paid LLM responses needed for training and evaluation but not for
playing a game.

No GPU anywhere: the listener is LightGBM and numpy, and
`LearnedListenerSpymaster.to_device` is a documented no-op. Measured on the
development machine, a turn costs 0.8 s and the process peaks at 1.7 GB, so
the real requirement is ~2 GB of free RAM.

Usage:
    python scripts/tools/make_demo_bundle.py --out ~/codenames-demo
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Everything the play path loads. Anything absent is skipped with a warning
# rather than failing: several of these are optional feature blocks, and a
# bundle missing one still plays (slightly worse) rather than not at all.
CACHE_FILES = [
    "similarity_tensor.npy", "similarity_meta.json", "clue_vocab.json",
    "board_vocab.json", "clue_stats.npz", "clue_stats_meta.json",
    "listener_gbt.txt", "word_stats.npz", "acronym_mask.npz",
    # Optional: the decoy-trained booster, so the game's spymaster menu can
    # offer the decoy variants. 25 MB; absent, those entries are just hidden.
    "listener_gbt_decoy.txt",
    "swow.npz", "entity_sims.npz", "lm_pmi.npz", "extra_sims.npz",
    "word_norms.npz", "wordnet_sims.npz", "lexical_sims.npz",
]

REQUIREMENTS = """\
# The play path only. The full project also needs gensim, wikiextractor,
# matplotlib and anthropic, none of which are imported to play a game.
numpy
scipy
lightgbm
wordfreq
# One call: torch.special.ndtr in codenames/spymasters/expected_words.py.
# scipy.special.ndtr would do the same job and is already required above, but
# swapping it changes float32 results in the last decimal, which would make
# this bundle disagree with every recorded result. Kept deliberately.
torch
"""

START_SH = """\
#!/usr/bin/env bash
# One command: make a venv if needed, install, serve, open a browser.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment (one-off, a few minutes -- torch is large)..."
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

echo "Starting. The first clue takes a second or two while the model loads."
exec ./.venv/bin/python play_server.py "$@"
"""

START_BAT = """\
@echo off
REM One command on Windows: make a venv if needed, install, serve.
cd /d "%~dp0"

if not exist .venv (
  echo Creating virtual environment ^(one-off, a few minutes -- torch is large^)...
  python -m venv .venv
  .venv\\Scripts\\pip install --quiet --upgrade pip
  .venv\\Scripts\\pip install --quiet -r requirements.txt
)

echo Starting. The first clue takes a second or two while the model loads.
.venv\\Scripts\\python play_server.py %*
"""

README = """\
# Codenames spymaster — playable demo

The real model, running locally. Boards are generated fresh each game, and
every clue is computed on the spot: nothing here is precomputed or canned.

## Run it

    ./start.sh            # macOS / Linux
    start.bat             # Windows

First run builds a virtual environment and installs dependencies, which takes
a few minutes (torch is a large download). After that it starts in seconds and
opens http://127.0.0.1:8000 automatically.

Options: `--port 9000` to move it, `--no-open` to skip the browser, `--no-k1`
to disable the k=1 similarity tiebreak.

## Choosing the spymaster

The game page has a menu: the incumbent, and the decoy-trained model at each
outside-option weight the arena sweep tested. The choice applies from the next
clue, so you can switch mid-game and compare.

## The blind study

Open http://127.0.0.1:8000/eval. You are Blue's guesser; each clue comes from
one of two spymasters chosen at random, and the page is never told which. Guess
as you would in a real game, and press **Stop** when you no longer see a
connection — that is the behaviour being measured. Every turn is appended to
`cache/human_eval.jsonl`; send that file back, or read it here with

    ./.venv/bin/python analyze_human_eval.py

Choose what is compared with `./start.sh --eval-arms incumbent,decoy_out25`.

## What it needs

- Python 3.11 or newer
- About 2 GB of free RAM (the model peaks at 1.7 GB)
- About 1.5 GB of disk once dependencies are installed
- **No GPU.** The listener is LightGBM and numpy; the GPU path is a no-op.

## How it works

Two models run per clue. A closed-form Gaussian scores all 11,145 legal clues
and shortlists 200; a gradient-boosted tree trained on ~9,000 LLM guesser
rankings rescores those, and an exponential-race calculation turns each
ranking into the expected reward of announcing that clue for k words. The clue
played is the argmax over both the word and the number.

The tree never sees which words belong to which team. It predicts only how a
guesser would rank the board, and the team split is applied afterwards in the
reward — a listener that knew the roles would look excellent and be worthless
as a model of a guesser.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    ap.add_argument("--model", type=Path, default=None,
                    help="booster to ship as listener_gbt.txt (default: the deployed one)")
    args = ap.parse_args()

    out = args.out.expanduser().resolve()
    (out / "cache").mkdir(parents=True, exist_ok=True)

    shutil.copytree(PROJECT_ROOT / "codenames", out / "codenames",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PROJECT_ROOT / "scripts" / "tools" / "play_server.py", out / "play_server.py")
    shutil.copy2(PROJECT_ROOT / "scripts" / "tools" / "analyze_human_eval.py",
                 out / "analyze_human_eval.py")
    shutil.copytree(PROJECT_ROOT / "scripts" / "tools" / "webplay", out / "webplay",
                    dirs_exist_ok=True)

    total, missing = 0, []
    for name in CACHE_FILES:
        src = args.cache_dir / name
        if name == "listener_gbt.txt" and args.model is not None:
            src = args.model
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, out / "cache" / name)
        total += src.stat().st_size

    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "README.md").write_text(README)
    (out / "start.sh").write_text(START_SH)
    (out / "start.sh").chmod(0o755)
    (out / "start.bat").write_text(START_BAT)

    print(f"bundle -> {out}")
    print(f"  cache: {total / 1e6:.0f} MB across {len(CACHE_FILES) - len(missing)} files")
    if missing:
        print(f"  MISSING (bundle still runs, with those features off): {missing}")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  total: {size / 1e6:.0f} MB")
    print(f"\nCopy that folder to the laptop, then: cd {out.name} && ./start.sh")


if __name__ == "__main__":
    main()
