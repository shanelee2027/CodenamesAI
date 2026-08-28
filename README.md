# CodenamesAI

A Codenames clue-giving agent built around a learned clue scorer over
multiple word embedding spaces. Codenames has two roles: the **codemaster**
sees which board words belong to which team and gives a one-word clue plus
a number; the **guesser** sees only the words and tries to pick their
team's words from the clue. This project builds the codemaster — the
guesser is deliberately simple and hand-written, a training environment
rather than a deliverable.

The project's initial design/build phase is done; it's now in an iteration
phase, trying ideas to improve the model. [`docs/versions/`](docs/versions/)
documents each model as it's built, historical ones once superseded.
[`docs/design-decisions.md`](docs/design-decisions.md) has the standing
design rationale (why the feature vector is built the way it is, why the
guesser pool looks like it does, what's explicitly out of scope), and
[`docs/log.md`](docs/log.md) is the day-to-day narrative of how any of this
was actually arrived at.

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

## How the pieces fit together

Work happens at three different times:

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
   happened, and train a model to predict that outcome from board+clue
   features.
   ```bash
   python scripts/generate_training_data.py --n-examples 200000
   python scripts/train_scorer.py --data-dir cache/training_data
   ```
3. **Play time (per turn).** Score every legal clue in the vocabulary in
   one batched forward pass, turn each score into an expected reward for
   every possible clue number, and return the best `(clue, number)` pair.

```bash
python scripts/web_inspector.py       # web UI: click a board, test any codemaster's clue choice,
                                       # simulate the resulting turn, adjust reward/noise/rarity live
python scripts/inspector.py           # CLI equivalent
python scripts/run_arena.py           # cross-play matrix: every codemaster x every guesser
```

`scripts/run_ablation_study.py` regenerates data and retrains a batch of
model variants at once (e.g. the noise-level sweep the current model doc
reports); see its own `--help` and docstring.

## Baselines

Fixed reference points, not iteration targets — the learned model (below)
should be compared against these, and should clearly beat them; if it
doesn't, something's wrong upstream, not a reason to add more to the model
itself.

1. **Random** — a random legal clue.
2. **Centroid** — the clue nearest the mean of a random own-word subset.
3. **Oracle** — zero-noise, single-space, deterministic "longest run of own
   words at the top of the ranking." Explicitly unrealistic (assumes a
   guesser with perfect, noise-free knowledge) — an exploration tool for an
   upper bound, not a real baseline.

(A fourth, `linear_scorer` — an 8-constant hand-coded formula, weighted
average across spaces then weighted sum across roles — exists in
`codenames/codemasters/linear_scorer.py` but isn't currently part of the
active comparison set.)

## Model 1: the (k, cause) scorer

The current (and, so far, only) learned model. Given a board state and a
candidate clue:

**The feature vector.** Look up the clue's similarity to all 25 board words
in every space, partition by role (own/opponent/neutral/assassin), sort
descending within each role group per space (order carries no information
otherwise, and sorting makes rank position meaningful), pad with a sentinel
and a validity mask, and concatenate every space's values plus a few
scalars (own words remaining, turn index, score differential) — ~103-128
dimensions depending on how many spaces are built. Spaces are concatenated,
never averaged — averaging would destroy exactly the "one space knows this,
another doesn't" signal the whole project exists to use. The model never
sees words, only these numbers; all linguistic knowledge lives in the
similarity tensor.

**The model.** An MLP (`codenames/scorer.py::Scorer`): ~103-128 inputs →
hidden layers (256, 256, 128) → 13 output classes → softmax. Those 13
classes are a joint distribution over `(k, cause)` — `k` is how many
own-words a guesser reveals in a row before stopping (0..3), crossed with
`cause`, which role actually stopped it (neutral / opponent / assassin),
plus one right-censored class for `k=4` (hit the cap, no miss). Naming
"cause," not just "k," is the model's central idea: a stop on an opponent
word and a stop on the assassin are very different outcomes, and this
model actually tells them apart instead of collapsing every non-own-word
stop into one undifferentiated "miss."

**Using the output.** The network's job stops at that distribution — it
never sees or is trained against a specific reward value. A separate
closed-form calculation (`codenames/scorer.py::reward_matrix`/
`expected_reward_and_best_n`) combines the predicted distribution with four
independent reward parameters (`own_reward`, `neutral_reward`,
`opponent_reward`, `assassin_reward`) to get an expected reward for every
possible announced number `n`, and returns whichever `(clue, n)` scores
highest. Because those four reward values live outside training entirely,
any of them — including `assassin_reward`, which doubles as a risk-aversion
knob — can be changed at play time with no retraining.

Full detail (attempts rule, the six trained noise-level checkpoints and
their results, what the web UI exposes on top of this, open questions for
the next model) is in [`docs/versions/v1.md`](docs/versions/v1.md).

## Layout

```
data/          raw dumps and embeddings (gitignored)
cache/         similarity tensor, generated datasets, checkpoints (gitignored)
codenames/     library code (board, similarity, features, guessers, codemasters, scorer, game, arena)
scripts/       pipeline scripts (build/generate/train) and the inspector/arena/web UI
tests/         pytest suite
docs/
  design-decisions.md   standing design rationale, not tied to one model
  log.md                chronological working log
  versions/             one doc per model, in order built
```

## Status

Corpus collection (Fandom dumps) and fastText training — the fourth
embedding space, meant to supply pop-culture/proper-noun knowledge — remain
unbuilt; the project currently runs on 3 of its 4 intended spaces (GloVe,
ConceptNet Numberbatch, Wikipedia2Vec). Human evaluation (logged games
against real players, fitting the guesser-pool mixture to that data) also
hasn't started. Everything else above is built and in active iteration —
see `docs/versions/v1.md` for the current model and `docs/log.md` for
what's actively being worked on.
