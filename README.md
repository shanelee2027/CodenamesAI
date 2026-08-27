# CodenamesAI

A Codenames clue-giving agent built around a learned clue scorer over
multiple word embedding spaces. See [`docs/SCOPE.md`](docs/SCOPE.md) for the
full design spec and [`CLAUDE.md`](CLAUDE.md) for the milestone checklist.

## Setup

Requires Python 3.11+ and, for GPU acceleration, a CUDA 12.8+-capable driver
(this project targets an RTX 5080 / Blackwell / sm_120, which needs CUDA
12.8+; the default PyPI `torch` wheel currently ships CUDA 13.0 and works
out of the box).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Verify the GPU is visible and working:

```bash
python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name())
a = torch.randn(2048, 2048, device='cuda')
print((a @ a).sum().item())
"
```

Run tests:

```bash
pytest
```

## Layout

```
data/          raw dumps and embeddings (gitignored)
cache/         similarity tensor, generated datasets, checkpoints (gitignored)
codenames/     library code (board, similarity, features, guessers, codemasters, scorer, game, arena)
scripts/       one-off and pipeline scripts
tests/         pytest suite
docs/          design spec (SCOPE.md) and working log (log.md)
```

## Status

Tracked in [`CLAUDE.md`](CLAUDE.md). Work proceeds one milestone (and one
module) at a time; expectations vs. outcomes are recorded in
[`docs/log.md`](docs/log.md).
