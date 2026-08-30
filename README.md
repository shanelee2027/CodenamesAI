# CodenamesAI

A Codenames clue-giving agent built around a learned clue scorer over
multiple word embedding spaces. Codenames has two roles: the **codemaster**
sees which board words belong to which team and gives a one-word clue plus
a number; the **guesser** sees only the words and tries to pick their
team's words from the clue. This project builds the codemaster — the
guesser is deliberately simple and hand-written, a training environment
rather than a deliverable.

[`docs/versions/`](docs/versions/) documents each model as it's built.
[`docs/design-decisions.md`](docs/design-decisions.md) has the standing
design rationale. [`docs/log.md`](docs/log.md) is the working log.

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
python scripts/web_inspector.py       # web UI: pick codemasters/guessers, play a full two-team
                                       # game, or inspect a single clue/turn with live reward/noise/rarity controls
python scripts/inspector.py           # CLI equivalent of the single-turn inspector
python scripts/run_arena.py           # single-team cross-play matrix: every codemaster x every guesser
python scripts/run_two_team_arena.py  # real two-team self-play, one codemaster+guesser pair on both sides
```

`scripts/run_ablation_study.py` regenerates data and retrains a batch of
model variants at once; see its own `--help` and docstring.

## Baselines

Fixed reference points, not iteration targets — the learned model (below)
should be compared against these, and should clearly beat them.

1. **Random** — a random legal clue.
2. **Centroid** — the clue nearest the mean of a random own-word subset.
3. **Oracle** — zero-noise, single-space, deterministic "longest run of own
   words at the top of the ranking." An exploration tool for an upper
   bound, not a realistic baseline.
4. **Linear scorer** (`linear_scorer`) — an 8-constant hand-coded formula,
   weighted average across spaces then weighted sum across roles.

## Evaluation

Two teams play a real game against each other (`codenames/game.py::play_two_team_game`):
the same codemaster+guesser pair on both sides, alternating turns, board
depleting from both sides' actual play. A symmetric win rate isn't
reported — since both teams run the identical codemaster/guesser, it
mostly reflects the fixed first-move edge (team A always has 9 words to
team B's 8), not model quality. Instead, both teams' turns are pooled
into one set of stats:

- **assassin-hit rate** — fraction of games ending in anyone hitting the
  assassin (`1 - clean-finish rate`).
- **half-turns** — one team's turn; a two-team game is naturally about 2x
  as long as a single-team one. Reported for all games and for
  clean-finish games only.
- **per-guess role breakdown** — of every word actually guessed, pooled
  across both teams: own / opponent / neutral / assassin.

`codenames/two_team_arena.py` runs this on CPU; `codenames/two_team_gpu_arena.py`
batches many simultaneous games on GPU for the same result, much faster.

## Model 1: the (k, cause) scorer

Given a board state and a candidate clue:

**The feature vector.** Look up the clue's similarity to all 25 board words
in every space, partition by role (own/opponent/neutral/assassin), sort
descending within each role group per space (order carries no information
otherwise, and sorting makes rank position meaningful), pad with a sentinel
and a validity mask, and concatenate every space's values plus a few
scalars (own words remaining, turn index, score differential). Spaces are
concatenated, never averaged — averaging would destroy exactly the "one
space knows this, another doesn't" signal the whole project exists to use.
The model never sees words, only these numbers; all linguistic knowledge
lives in the similarity tensor.

**The model.** An MLP (`codenames/scorer.py::Scorer`): inputs → hidden
layers (256, 256, 128) → 13 output classes → softmax. Those 13 classes are
a joint distribution over `(k, cause)` — `k` is how many own-words a
guesser reveals in a row before stopping (0..3), crossed with `cause`,
which role actually stopped it (neutral / opponent / assassin), plus one
right-censored class for `k=4` (hit the cap, no miss).

**Using the output.** The network predicts that distribution only — it's
never trained against a specific reward value. A separate closed-form
calculation (`codenames/scorer.py::reward_matrix`/`expected_reward_and_best_n`)
combines the predicted distribution with four independent reward
parameters (`own_reward`, `neutral_reward`, `opponent_reward`,
`assassin_reward`) to get an expected reward for every possible announced
number `n`, and returns whichever `(clue, n)` scores highest. Those four
reward values live outside training entirely, so any of them — including
`assassin_reward`, which doubles as a risk-aversion knob — can be changed
at play time with no retraining.

**Results.** Real two-team self-play, `noise_std=0.08`'s `noisy_glove`
guesser, 300 boards:

| codemaster | assassin-hit rate | half-turns (all) | half-turns (clean) |
|---|---|---|---|
| **model 1 (learned)** | **0.0%** | **9.03** | **9.03** |
| centroid | 13.3% | 10.33 | 11.03 |
| linear_scorer | 5.0% | 18.69 | 18.69 |
| random | 83.3% | 9.77 | 15.30 |

| codemaster | own | opponent | neutral | assassin |
|---|---|---|---|---|
| **model 1 (learned)** | **97.4%** | **0.3%** | **2.3%** | **0.0%** |
| centroid | 90.0% | 4.1% | 4.9% | 0.9% |
| linear_scorer | 40.5% | 31.4% | 27.8% | 0.2% |
| random | 32.8% | 33.3% | 27.3% | 6.5% |

Full detail (all six trained noise-level checkpoints, what the web UI
exposes on top of this, open questions for the next model) is in
[`docs/versions/v1.md`](docs/versions/v1.md).

### Model 1.1: same scorer, a blended guesser

A subversion, not a new architecture — identical `(k, cause)` scorer and
feature vector as model 1, retrained against a different guesser pool:
instead of 3 separate noisy single-space guessers, a single `BlendGuesser`
(weighted average of cosine similarity across all three spaces — glove
0.3, numberbatch 0.5, wikipedia2vec 0.2 — plus `noise_std=0.08`). This is
a deliberate departure from [`docs/design-decisions.md`](docs/design-decisions.md)'s
"diversity must be in knowledge, not noise" principle: there's one
(synthetic) listener here, not three differently-knowledgeable ones.
Available in the web UI as `learned:blend`, and as a guesser choice
(`blend`) alongside the standard 3.

**Results.** Real two-team self-play, 300 boards, model 1.1 vs. its own
blend guesser on both sides (baselines evaluated with the same guesser —
not a controlled comparison against model 1, which uses a different,
harder guesser pool):

| codemaster | assassin-hit rate | half-turns (all) | half-turns (clean) |
|---|---|---|---|
| **model 1.1 (learned)** | **5.0%** | **8.18** | **8.35** |
| centroid | 17.0% | 10.64 | 11.62 |
| linear_scorer | 8.7% | 18.97 | 19.21 |
| random | 85.0% | 9.26 | 15.33 |

| codemaster | own | opponent | neutral | assassin |
|---|---|---|---|---|
| **model 1.1 (learned)** | **86.9%** | **3.6%** | **9.2%** | **0.3%** |
| centroid | 86.0% | 5.6% | 7.2% | 1.1% |
| linear_scorer | 39.3% | 32.1% | 28.2% | 0.4% |
| random | 32.3% | 34.3% | 26.3% | 7.1% |

See [`docs/versions/v1.1.md`](docs/versions/v1.1.md) for run details.

### History-aware guessers

`HistoryAwareGuesser` (`codenames/guessers/history_aware.py`) can spend one
earned bonus guess per turn — real Codenames' `n+1` rule, reinstated only
when a past clue's miss plausibly left a word unaccounted-for. It requires
no codemaster or training changes. In real two-team self-play (300 boards,
each model's own guesser with vs. without history-awareness), it makes
both models measurably worse:

| pairing | assassin-hit rate | own rate |
|---|---|---|
| model 1 + noisy_glove | 0.0% | 97.4% |
| model 1 + history_aware_noisy_glove | 7.3% | 85.2% |
| model 1.1 + blend | 5.0% | 86.9% |
| model 1.1 + history_aware_blend | 7.3% | 83.8% |

Not adopted — see [`docs/versions/v1.md`](docs/versions/v1.md)'s open
questions for why.

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
