# CodenamesAI

A Codenames spymaster that never asks a language model for a clue. Codenames
has two roles: the **spymaster** sees which board words belong to which team
and gives a one-word clue plus a number; the **guesser** sees only the words
and tries to pick their team's. This project builds the spymaster.

It works by modelling the guesser. A gradient-boosted listener, distilled from
an LLM guesser's rankings, predicts which board word a guesser will pick for a
clue; the spymaster then searches the clue vocabulary for the (clue, number)
pair with the highest expected reward under that listener. LLMs appear only as
the teacher the listener is distilled from and as the guesser that evaluation
plays against.

- [`docs/worklog.md`](docs/worklog.md): what has been tried, with results.
  Start here.
- [`docs/design-decisions.md`](docs/design-decisions.md): standing rationale.
- [`docs/iteration-architecture.md`](docs/iteration-architecture.md): how a
  model is registered, trained and evaluated.
- [`docs/versions/`](docs/versions/): one doc per model.
- [`docs/log.md`](docs/log.md): the working log, entry by entry.

## Setup

Python 3.11+. The listener, play server and arenas are CPU-only; the GPU
(an RTX 5080 here) is used only to build language-model features.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # scripts import `codenames` through this install
pytest
```

API keys (`ANTHROPIC_API_KEY`, `DEEPINFRA_API_KEY`) go in a gitignored `.env`
at the repo root (see `.env.example`), never in the shell profile: Claude Code
reads the same variable.

## The models

| Model | What it is | Role |
|---|---|---|
| `centroid` | clue nearest the mean of a random own-word subset; no penalty for other words | reference opponent |
| `expected_words` | guesser = similarity z-score + Gaussian noise N(0, σ²); exact expected reward by conditioning on the strongest distractor ([paper](docs/clue-selection-theory.pdf), [doc](docs/versions/expected_words.md)) | baseline, and stage one of the incumbent |
| `learned_listener` | shortlists 200 clues with `expected_words` (σ=1.5), then scores them with the distilled listener's exact Plackett–Luce expected reward ([paper](docs/clue-selection-learned.pdf), [doc](docs/versions/learned_listener.md)) | **the incumbent** |

All three are entries in `configs/spymasters.json`; a script picks spymasters
by registry name, never by import.

## How the pieces fit together

1. **Build time, once.** Pretrained embeddings (GloVe, ConceptNet
   Numberbatch, Wikipedia2Vec) become one similarity tensor
   `[clues × board words × spaces]`, memory-mapped; the models are never
   loaded again. Per-clue statistics turn a raw similarity into a z-score,
   and side tables add the listener's other evidence: word association
   norms (SWOW), language-model PMI, extra embedding spaces, concreteness
   and WordNet.
   ```bash
   python scripts/data/build_similarity_tensor.py
   python scripts/data/build_clue_stats.py
   ```
2. **Teacher data.** An LLM guesser ranks the board for sampled clues;
   every response is cached in `cache/llm_store.db`.
   ```bash
   python scripts/data/collect_listener_data.py     # rankings
   python scripts/data/collect_decoy_data.py        # rankings with off-board decoys
   ```
3. **Train the listener** (LightGBM, group softmax, split by board seed).
   ```bash
   python scripts/pipeline/train_listener.py --decoys cache/training_data/decoys.jsonl
   ```
4. **Play.** `scripts/tools/play_server.py` serves a local game against any
   registered spymaster, and `/eval` runs the blind one-clue human study.
5. **Evaluate** head to head (below).

## Guessers

A guesser is named by one spec string
(`codenames/guessers/registry.py::build_guesser`):

| Spec | Guesser |
|---|---|
| `anthropic:claude-sonnet-5:medium` | the evaluation guesser |
| `deepinfra:openai/gpt-oss-120b` | the teacher, and the cheap guesser for sweeps (effort low) |
| `anthropic:claude-opus-5:medium` | reserved for final headline numbers |
| `noisy_glove` etc. | synthetic embedding guessers from `configs/guesser_pool.json`: free, and weak evidence |

## Evaluation

Every comparison is a **head-to-head matchup**: two spymasters, one per side,
sharing one guesser, with every board played twice and the sides swapped,
because team A holds 9 words and moves first. The unit is the board, and the
headline is the sign test on boards one side won both ways.

- **Sweeps** use gpt-oss on fresh seeds (`scripts/tools/sweep_role_costs.py`,
  `scripts/tools/analyze_headtohead.py`). Comparable within one sweep.
- **The frozen suite** `holdout_v1` (`configs/eval_suite.json`) is the
  constant benchmark: 100 fixed boards drawn only from 150 held-out words no
  model trained on, against Sonnet. Games are stored per (pair, seating,
  suite, board), so a pair is paid for once and extending the suite plays
  only the new boards.
  ```bash
  python scripts/pipeline/run_eval_suite.py learned_listener expected_words \
      --opponent-param sigma=1.5 --dry-run
  ```

**Results so far** (Sonnet guessing, fresh seeds, not the frozen suite):
`learned_listener` beat `expected_words` at σ=1.5 in 72% of 100 games
(95% CI 0.63–0.80), `centroid` in 82% of 40 and `expected_words` at σ=2.5 in
90% of 40. The frozen suite has not been run yet. Full numbers are in the
[worklog](docs/worklog.md).

## Layout

```
codenames/     library: board, similarity, guessers, spymasters, listener
               features and training, pl_reward, game, arenas, eval suite
configs/       spymaster registry, synthetic guesser pool, frozen eval suite
scripts/
  data/        build time and teacher-data collection
  pipeline/    train the listener, run arenas, run the eval suite
  tools/       analysis, sweeps, play server, exports
notebooks/     arena results, listener SHAP
scratch/       exploratory notebooks
tests/         pytest suite
docs/          worklog, design, iteration architecture, papers, log
cache/, data/  generated artifacts and raw data (gitignored)
```
