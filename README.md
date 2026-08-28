# CodenamesAI

A Codenames clue-giving agent built around a learned clue scorer over
multiple word embedding spaces. Codenames has two roles: the **codemaster**
sees which board words belong to which team and gives a one-word clue plus
a number; the **guesser** sees only the words and tries to pick their
team's words from the clue. This project builds the codemaster — the
guesser is deliberately simple and hand-written, a training environment
rather than a deliverable.

The project's initial design/build phase is done; it's now in an iteration
phase, improving the model version over version. See
[`docs/versions/`](docs/versions/) for what's changed between versions and
[`docs/design-decisions.md`](docs/design-decisions.md) for the standing
design rationale (why the feature vector is built the way it is, why the
guesser pool looks like it does, what's explicitly out of scope). The
day-to-day narrative of how any of this was actually arrived at is in
[`docs/log.md`](docs/log.md).

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

## How the model works

**Three stages, at three different times:**

1. **Build time (once).** Download/prepare each embedding space, fix a
   clue vocabulary and a board vocabulary, and precompute a similarity
   tensor `[n_clues × n_board_words × n_spaces]` (fp16, memory-mapped).
   After this, the embedding models themselves are never loaded again —
   everything downstream reads the tensor.
   ```bash
   python scripts/build_similarity_tensor.py
   python scripts/extend_similarity_tensor.py   # add a space to an existing tensor
   ```
2. **Train time (repeatable).** Simulate many sampled (board, clue,
   guesser) triples against the guesser pool, label each with what actually
   happened, and train an MLP to predict that outcome from board+clue
   features.
   ```bash
   python scripts/generate_training_data.py --n-examples 200000
   python scripts/train_scorer.py --data-dir cache/training_data
   ```
3. **Play time (per turn).** Score every legal clue in the vocabulary in
   one batched forward pass, turn each score into an expected reward for
   every possible clue number, and return the best `(clue, number)` pair.

**The feature vector**, given a board state and a candidate clue: look up
the clue's similarity to all 25 board words in every space, partition by
role (own/opponent/neutral/assassin), sort descending within each role
group per space (order carries no information otherwise, and sorting makes
rank position meaningful), pad with a sentinel and a validity mask, and
concatenate every space's values plus a few scalars (own words remaining,
turn index, score differential) — ~103-128 dimensions depending on how many
spaces are built. Spaces are concatenated, never averaged — averaging would
destroy exactly the "one space knows this, another doesn't" signal the
whole project exists to use. The model never sees words, only these
numbers; all linguistic knowledge lives in the similarity tensor.

**The model** is an MLP (`codenames/scorer.py::Scorer`): ~103-128 inputs →
hidden layers (256, 256, 128) → some number of output classes → softmax.
What those output classes mean, and how they turn into a `(clue, number)`
decision, is version-specific — see [`docs/versions/`](docs/versions/) for
the current version's exact output shape and scoring formula. In broad
strokes: the model predicts a distribution over how a turn using a given
clue would play out, and a separate closed-form calculation (not the
network) turns that distribution into an expected reward for every
possible announced number, letting risk tolerance be adjusted at *scoring*
time with no retraining.

## Testing and inspecting a model

```bash
python scripts/web_inspector.py       # web UI: click a board, test any codemaster's clue choice,
                                       # simulate the resulting turn, adjust reward/noise/rarity live
python scripts/inspector.py           # CLI equivalent
python scripts/run_arena.py           # cross-play matrix: every codemaster x every guesser
```

`scripts/run_ablation_study.py` regenerates data and retrains a batch of
model variants at once (e.g. the noise-level sweep each version doc
reports); see its own `--help` and docstring.

## Baselines

Fixed reference points, not iteration targets — every real model version
should be compared against these, not just against the previous version:

1. **Random** — a random legal clue.
2. **Centroid** — the clue nearest the mean of a random own-word subset.
3. **Linear scorer** — an 8-constant hand-coded formula (weighted average
   across spaces, then weighted sum across roles), untuned illustrative
   constants.
4. **Oracle** — zero-noise, single-space, deterministic "longest run of own
   words at the top of the ranking." Explicitly unrealistic (assumes a
   guesser with perfect, noise-free knowledge) — an exploration tool for an
   upper bound, not a real baseline.

The learned model (see `docs/versions/`) should clearly beat all of these;
if it doesn't, something's wrong upstream, not a reason to add more to the
model itself.

## Layout

```
data/          raw dumps and embeddings (gitignored)
cache/         similarity tensor, generated datasets, checkpoints (gitignored)
codenames/     library code (board, similarity, features, guessers, codemasters, scorer, game, arena)
scripts/       pipeline scripts (build/generate/train) and the inspector/arena/web UI
tests/         pytest suite
docs/
  design-decisions.md   standing design rationale, not tied to one version
  log.md                chronological working log
  versions/             what changed between model versions
```

## Status

Corpus collection (Fandom dumps) and fastText training — the fourth
embedding space, meant to supply pop-culture/proper-noun knowledge — remain
unbuilt; the project currently runs on 3 of its 4 intended spaces (GloVe,
ConceptNet Numberbatch, Wikipedia2Vec). Human evaluation (logged games
against real players, fitting the guesser-pool mixture to that data) also
hasn't started. Everything else in "How the model works" above is built and
in active iteration — see `docs/versions/` for the current version and
`docs/log.md` for what's actively being worked on.
