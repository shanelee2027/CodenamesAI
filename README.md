# CodenamesAI

A Codenames clue-giving agent built around a learned clue scorer over
multiple word embedding spaces. Codenames has two roles: the **spymaster**
sees which board words belong to which team and gives a one-word clue plus
a number; the **guesser** sees only the words and tries to pick their
team's words from the clue. This project builds the spymaster — the
guesser is deliberately simple and hand-written, a training environment
rather than a deliverable.

[`docs/versions/`](docs/versions/) documents each model as it is built (empty
for now — no model has been built under the current eval suite).
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
   python scripts/data/build_similarity_tensor.py
   python scripts/data/extend_similarity_tensor.py   # add a space to an existing tensor
   ```
2. **Train time (repeatable).** Simulate many sampled (board, clue,
   guesser) triples against the guesser pool, label each with what actually
   happened, and train a model to predict that outcome from board+clue
   features.
   Split in two, so the expensive half is paid once: generation simulates
   guesser *rollouts* (model-independent, the dominant cost), and
   featurization turns those into a model's feature vectors (cheap, ~35x
   faster). A new feature design re-featurizes stored rollouts instead of
   re-simulating them — see `codenames/rollouts.py` and
   [`docs/iteration-architecture.md`](docs/iteration-architecture.md).
   ```bash
   python scripts/pipeline/generate_training_data.py --n-examples 200000   # -> cache/rollouts
   python scripts/pipeline/featurize_rollouts.py                           # -> cache/training_data
   python scripts/pipeline/train_scorer.py --data-dir cache/training_data
   ```
3. **Play time (per turn).** Score every legal clue in the vocabulary in
   one batched forward pass, turn each score into an expected reward for
   every possible clue number, and return the best `(clue, number)` pair.

```bash
python scripts/tools/web_inspector.py       # web UI: pick spymasters/guessers, play a full two-team
                                       # game, or inspect a single clue/turn with live reward/noise/rarity controls
python scripts/tools/inspector.py           # CLI equivalent of the single-turn inspector
python scripts/pipeline/run_arena.py           # single-team cross-play matrix: every spymaster x every guesser
python scripts/pipeline/run_two_team_arena.py  # real two-team self-play, one spymaster+guesser pair on both sides
```

`scripts/pipeline/run_ablation_study.py` regenerates data and retrains a batch of
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
the same spymaster+guesser pair on both sides, alternating turns, board
depleting from both sides' actual play. A symmetric win rate isn't
reported — since both teams run the identical spymaster/guesser, it
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

## The learned scorer: (k, cause)

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

**Results.** None published. The two models that had measured results
(`v1` and the `v1.1` blend subversion) were retired: they were trained
against the old 60-word board holdout, so their numbers are not comparable
with anything trained under the current 150-word split, and reporting them
beside a new model would flatter them. Their measurements remain in
[`docs/log.md`](docs/log.md).

The evaluation that replaces them is defined in
[`docs/iteration-architecture.md`](docs/iteration-architecture.md): a frozen
suite of board seeds built entirely from held-out words, played against a
single fixed LLM guesser (Claude Opus 5 at medium effort), with every game
cached so a model is paid for exactly once. No model has been run against
it yet.

### Guessers that exist but aren't the training default

Two guessers were built, measured, and left in the tree rather than
adopted. Both are training-side infrastructure, not models:

- **`BlendGuesser`** — one weighted average of cosine similarity across all
  three spaces, rather than three separate single-space guessers. A
  deliberate departure from [`docs/design-decisions.md`](docs/design-decisions.md)'s
  "diversity must be in knowledge, not noise" principle, since it presents
  one synthetic listener instead of three differently-knowledgeable ones.
- **`HistoryAwareGuesser`** — spends one earned bonus guess per turn (real
  Codenames' `n+1` rule), reinstated only when a past clue's miss plausibly
  left a word unaccounted for. Measured worse for both models that were
  tested against it, on both assassin-hit rate and own-word rate.

Both remain selectable in the web UI and in `configs/guesser_pool_*.json`.
See [`docs/log.md`](docs/log.md) for the numbers behind each.

## Layout

```
data/          raw dumps and embeddings (gitignored)
cache/         similarity tensor, rollouts, datasets, checkpoints (gitignored)
codenames/     library code (board, similarity, features, guessers, spymasters,
               scorer, rollouts, eval_suite, game, arena)
scripts/
  data/        build time, run once: downloads, corpus, similarity tensor
  pipeline/    the iteration loop: generate, featurize, train, arenas
  tools/       interactive: inspector, web UI, record dumps
tests/         pytest suite
docs/
  design-decisions.md        standing design rationale, not tied to one model
  iteration-architecture.md  how building/training/evaluating a model works
  log.md                     chronological working log
  versions/                  one doc per model (empty: no model yet under the
                             current eval suite)
```

## Status

Corpus collection (Fandom dumps) and fastText training — the fourth
embedding space, meant to supply pop-culture/proper-noun knowledge — remain
unbuilt; the project currently runs on 3 of its 4 intended spaces (GloVe,
ConceptNet Numberbatch, Wikipedia2Vec). Human evaluation (logged games
against real players, fitting the guesser-pool mixture to that data) also
hasn't started.

The infrastructure above is built: the spymaster registry, the rollout
cache, the frozen eval suite and the eval store all work and are tested.
**No model currently holds published results** — the earlier ones were
retired along with the 60-word holdout they were trained against, and the
next model is the first to be built and evaluated under the current setup.
See [`docs/iteration-architecture.md`](docs/iteration-architecture.md) for
how that pipeline fits together and [`docs/log.md`](docs/log.md) for what's
actively being worked on.
