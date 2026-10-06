"""Pack a self-contained folder that plays the real model on any laptop.

The point is a demo that survives being moved: `cache/` is gitignored, so a
fresh clone has the code and none of the artifacts, and rebuilding them means
re-downloading several embedding sets. This copies exactly what the play
server reads and nothing else -- notably NOT `cache/llm_store.db`, which is
78 MB of paid LLM responses needed for training and evaluation but not for
playing a game.

No GPU anywhere: the listener is LightGBM and numpy. Measured on the
development machine, a turn costs 0.8 s and the process peaks at 1.7 GB, so
the real requirement is ~2 GB of free RAM.

**The bundle mirrors the repo's layout** (`codenames/`, `scripts/tools/`,
`cache/`), so the same zip works two ways: unpacked on its own, or unpacked
into a clone of the repo, where its code lands on the same paths and its
`cache/` fills the gitignored one. Its own files have names the repo does not
use (DEMO.md, requirements-demo.txt, start.sh, start.bat), and .gitignore
lists them, so a clone stays clean. In a clone newer than the bundle, extract
only `cache/`, or the bundle's older code would overwrite the clone's.

Usage:
    python scripts/tools/make_demo_bundle.py          # -> cache/codenames-demo/ and cache/codenames-demo.zip
"""

from __future__ import annotations

import argparse
import shutil
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
    # Optional: the association-trained booster and its rate link, for the
    # assoc / assoc_pass / assoc_pass_strong entries. Both or neither.
    "listener_gbt_assoc_w0.3.txt", "listener_gbt_assoc_w0.3.assoc.json",
    "swow.npz", "entity_sims.npz", "lm_pmi.npz", "extra_sims.npz",
    "word_norms.npz", "wordnet_sims.npz", "lexical_sims.npz",
    # Optional: the stop listeners (stop_listener, stop_net_words_listener;
    # both label sets), with the assoc profile booster their search runs on.
    "listener_gbt_assoc_profile.txt", "listener_gbt_stop.txt", "listener_gbt_stop_guess.txt",
    "assoc_profile.npz", "assoc_sims.npz", "isa_sims.npz", "conceptnet_sims.npz", "win_value.npz",
]

REQUIREMENTS = """\
# The play path only. The full project also needs gensim, matplotlib,
# anthropic and openai, none of which are imported to play a game.
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
  ./.venv/bin/pip install --quiet -r requirements-demo.txt
fi

echo "Starting. The first clue takes a second or two while the model loads."
exec ./.venv/bin/python scripts/tools/play_server.py "$@"
"""

START_BAT = """\
@echo off
REM One command on Windows: make a venv if needed, install, serve.
cd /d "%~dp0"

if not exist .venv (
  echo Creating virtual environment ^(one-off, a few minutes -- torch is large^)...
  python -m venv .venv
  .venv\\Scripts\\pip install --quiet --upgrade pip
  .venv\\Scripts\\pip install --quiet -r requirements-demo.txt
)

echo Starting. The first clue takes a second or two while the model loads.
.venv\\Scripts\\python scripts\\tools\\play_server.py %*
"""

README = """\
# Codenames spymaster — playable demo

The real model, running locally. Boards are generated fresh each game, and
every clue is computed on the spot: nothing here is precomputed or canned.

## Run it

    ./start.sh            # macOS / Linux
    start.bat             # Windows

## Or inside a clone of the repo

The bundle has the repo's layout. Unpack it into the clone's root: its
`cache/` fills the clone's (gitignored) `cache/`, its code lands on the same
paths, and its own files (this one, requirements-demo.txt, start.sh,
start.bat) are gitignored, so `git status` stays clean. Then run `./start.sh`,
or `python scripts/tools/play_server.py` from an environment that has the
project installed (`pip install -e .`).

If the clone is newer than the bundle, take only the data, so the bundle's
older code does not overwrite it:

    unzip codenames-demo.zip 'cache/*' -d path/to/CodenamesAI

First run builds a virtual environment and installs dependencies, which takes
a few minutes (torch is a large download). After that it starts in seconds and
opens http://127.0.0.1:8000 automatically.

Options: `--port 9000` to move it, `--no-open` to skip the browser, `--no-k1`
to disable the k=1 similarity tiebreak.

## Choosing the spymaster

The game page has a menu of every model whose files are in `cache/`: the
incumbent, the decoy-trained variants, and the four stop listeners ("Stop
listener" and "Stop listener, strict", each by chance of winning or by net
words). The stop listeners take 3-4 s a clue. The choice applies from the
next clue, so you can switch mid-game and compare.

## The blind study

Open http://127.0.0.1:8000/eval. You are Blue's guesser; each clue comes from
one of two spymasters chosen at random, and the page is never told which. Guess
as you would in a real game, and press **Stop** when you no longer see a
connection — that is the behaviour being measured. Every turn is appended to
`cache/human_eval.jsonl`; send that file back, or read it here with

    ./.venv/bin/python scripts/tools/analyze_human_eval.py

Choose what is compared with `./start.sh --eval-arms incumbent,decoy_out25`.

## Comparing two spymasters

Open http://127.0.0.1:8000/compare. Pick two models and deal: it finds a
position where their clues differ and shows both, with the words each clue is
meant for marked on the board. Vote which is better (keys 1-4, N to deal).
Votes are appended to `cache/compare_votes.jsonl`; send that file back.

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
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "cache" / "codenames-demo",
                    help="the folder; a .zip of it is written beside it")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    ap.add_argument("--model", type=Path, default=None,
                    help="booster to ship as listener_gbt.txt (default: the deployed one)")
    args = ap.parse_args()

    out = args.out.expanduser().resolve()
    (out / "cache").mkdir(parents=True, exist_ok=True)

    shutil.copytree(PROJECT_ROOT / "codenames", out / "codenames",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tools = out / "scripts" / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "tools" / "play_server.py", tools / "play_server.py")
    shutil.copy2(PROJECT_ROOT / "scripts" / "tools" / "analyze_human_eval.py", tools / "analyze_human_eval.py")
    shutil.copytree(PROJECT_ROOT / "scripts" / "tools" / "webplay", tools / "webplay", dirs_exist_ok=True)

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

    (out / "requirements-demo.txt").write_text(REQUIREMENTS)
    (out / "DEMO.md").write_text(README)
    (out / "start.sh").write_text(START_SH)
    (out / "start.sh").chmod(0o755)
    (out / "start.bat").write_text(START_BAT)

    print(f"bundle -> {out}")
    print(f"  cache: {total / 1e6:.0f} MB across {len(CACHE_FILES) - len(missing)} files")
    if missing:
        print(f"  MISSING (bundle still runs, with those features off): {missing}")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  total: {size / 1e6:.0f} MB")
    z = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"  zip -> {z} ({Path(z).stat().st_size / 1e6:.0f} MB)")
    print(f"\nCopy the zip to the laptop and unpack it, on its own or into a clone (DEMO.md), then ./start.sh")


if __name__ == "__main__":
    main()
