# CodenamesAI

A Codenames clue-giving agent that picks clues by a closed-form scoring
function over word-embedding similarities — no language model produces a
clue at any point. Codenames has two roles: the **spymaster**
sees which board words belong to which team and gives a one-word clue plus
a number; the **guesser** sees only the words and tries to pick their
team's words from the clue. This project builds the spymaster — the
guesser is deliberately simple and hand-written, a training environment
rather than a deliverable.

[`docs/versions/`](docs/versions/) documents each model as it is built; the
current one is [`expected_words`](docs/versions/expected_words.md).
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
2. **Per-clue statistics (once).** Mean and standard deviation of each
   clue's similarity across all 400 board words, plus a rarity
   percentile, so a raw similarity can be turned into a per-clue z-score.
   ```bash
   python scripts/data/build_clue_stats.py   # -> cache/clue_stats.npz
   ```
3. **Play time (per turn).** Score every legal clue in the vocabulary
   against the current board, pick the best `(clue, number)` pair by
   expected reward. Nothing is trained; the scoring function is written
   down, not fitted.

```bash
python scripts/pipeline/run_arena.py           # single-team cross-play matrix: every spymaster x every guesser
python scripts/pipeline/run_two_team_arena.py  # real two-team self-play, one spymaster+guesser pair on both sides
```

Both are **training diagnostics against synthetic guessers, not
scoreboards** — evaluation results come only from the frozen LLM eval
suite (`codenames/eval_suite.py`).

## Baselines

Fixed reference points, not iteration targets — a new model should be
compared against these, and should clearly beat them.

1. **Random** — a random legal clue.
2. **Centroid** — the clue nearest the mean of a random own-word subset.
3. **Oracle** — zero-noise, single-space, deterministic "longest run of own
   words at the top of the ranking." An exploration tool for an upper
   bound, not a realistic baseline.
4. **Linear scorer** (`linear_scorer`) — an 8-constant hand-coded formula,
   weighted average across spaces then weighted sum across roles.
5. **Expected words** (`expected_words`) — see
   [`docs/versions/expected_words.md`](docs/versions/expected_words.md).
   Threshold-free: z-scores each candidate clue against every unrevealed
   board word (`cache/clue_stats.npz`, built by
   `scripts/data/build_clue_stats.py`) and jointly picks the clue and
   number that maximize a single metric — a sub-linear-in-k expected-
   words-reached term minus a risk term weighted by each role's real
   reward magnitude. Replaces the earlier `z_threshold` baseline, whose
   hard role thresholds needed a fallback chain this metric has no need
   for.

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

## The current model: `expected_words`

Given a board state and a candidate clue, in one embedding space
(`numberbatch`), with every similarity converted to a per-clue z-score:

```
q_w(x, tau) = Phi((b_w - x) / tau)          # b_w = z-score of non-own word w
s_j         = prod_w (1 - q_w(a_j, tau_gain))
gain(k)     = sum_{j=1..k} prod_{i<=j} s_i
penalty(k)  = sum_w c_w * q_w(a_k, tau_pen)
score       = gain(k) - penalty(k)
```

`a_1 >= a_2 >= ...` are the clue's z-scores against unrevealed own words
and `c_w` is the cost of revealing `w`. `gain(k)` is deliberately
sub-linear: the k-th word is credited only by the probability the guesser
survives that far, which is what makes `k` a real choice rather than
always maxing out. The selected `(clue, number)` is the joint argmax.

Nothing is trained and nothing is fitted — the two constants
(`tau_gain=2.5`, `tau_pen=0.7`) are hand-set and have never been swept.
Full derivation, the rejected linear variant, and measurements are in
[`docs/versions/expected_words.md`](docs/versions/expected_words.md).

The underlying order-statistics model — what the guesser's ranking looks
like when their embedding disagrees with ours by Gaussian noise — is
derived in [`docs/clue-selection-theory.pdf`](docs/clue-selection-theory.pdf).

**Results.** None published. The two models that had measured results
(`v1` and the `v1.1` blend subversion) were retired: they were trained
against the old 60-word board holdout, so their numbers are not comparable
with anything under the current 150-word split, and reporting them beside
a new model would flatter them. Their measurements remain in
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
  three spaces, rather than three separate single-space guessers. It
  presents one synthetic listener instead of three differently-
  knowledgeable ones, which is the opposite of what a training pool is
  for.
- **`HistoryAwareGuesser`** — spends one earned bonus guess per turn (real
  Codenames' `n+1` rule), reinstated only when a past clue's miss plausibly
  left a word unaccounted for. Measured worse for both models that were
  tested against it, on both assassin-hit rate and own-word rate.

Both remain selectable in `configs/guesser_pool_*.json`.
See [`docs/log.md`](docs/log.md) for the numbers behind each.

## Layout

```
data/          raw dumps and embeddings (gitignored)
cache/         similarity tensor, rollouts, datasets, checkpoints (gitignored)
codenames/     library code (board, similarity, features, guessers, spymasters,
               scorer, rollouts, eval_suite, game, arena)
scripts/
  data/        build time, run once: downloads, similarity tensor
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

The project runs on 3 embedding spaces (GloVe, ConceptNet Numberbatch,
Wikipedia2Vec); a fourth, pop-culture/proper-noun space was planned but is
now out of scope. Human evaluation (logged games against real players,
fitting the guesser-pool mixture to that data) also hasn't started.

The infrastructure above is built: the spymaster registry, the rollout
cache, the frozen eval suite and the eval store all work and are tested.
**No model currently holds published results** — the earlier ones were
retired along with the 60-word holdout they were trained against, and the
next model is the first to be built and evaluated under the current setup.
See [`docs/iteration-architecture.md`](docs/iteration-architecture.md) for
how that pipeline fits together and [`docs/log.md`](docs/log.md) for what's
actively being worked on.
