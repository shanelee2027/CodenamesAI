# Design decisions

Standing design rationale that doesn't belong in the README's overview and
isn't tied to one model version. Recorded so it isn't silently revisited.
See [`docs/versions/`](versions/) for what changed between models,
[`docs/iteration-architecture.md`](iteration-architecture.md) for how a
model gets built and evaluated, and [`docs/log.md`](log.md) for the
chronological record of how any of this was actually arrived at —
including the rationale for approaches since retired.

## What this project explicitly is not

- **Not an LLM wrapper.** No language model is prompted for a clue
  anywhere in the pipeline. This is the project's defining constraint,
  not a performance claim.
- **Not novel research.** Reusing published techniques is fine and
  expected; the contribution is a working, well-measured system.
- **Not a training-from-scratch embeddings project.** Every embedding
  space is downloaded pretrained.

## Every model is reported against a baseline ladder

`configs/spymasters.json` registers four non-learned spymasters with
`"roles": ["baseline"]` — `random`, `centroid`, `linear_scorer`, and the
current `expected_words` — and any new model picks all of them up
without naming them (`codenames/spymasters/registry.py`).

The overhead is deliberate. A win rate with nothing beside it is not a
result, and the cheapest way to find out that an elaborate model is
doing nothing is to see it tied with `centroid`. `random` earns its slot
separately: it catches harness bugs that would otherwise make every
model look equally good.

A baseline is retired by replacement, not by accumulation — when
`expected_words` superseded `z_threshold`, the older model's code, test,
config entry, and doc were deleted rather than kept side by side, with
its measurements preserved in `docs/log.md`. See
[`versions/expected_words.md`](versions/expected_words.md).

## Two structural guards against overfitting

Both are enforced by code and config rather than by discipline, which is
the point — neither can be violated by forgetting.

**Held-out board words.** `codenames/assets/board_words_holdout.txt`
holds 150 of the 400 board words out of training data generation
entirely (`codenames/board.py::load_training_wordlist()`), so evaluation
builds boards from words no model has seen. This tests generalization to
unseen board *content*, which is a different axis from generalizing to an
unseen listener. See `docs/iteration-architecture.md` step 5 for why 150
rather than the original 60 — the change is what retired `v1`/`v1.1`,
whose numbers are not comparable with anything trained since.

**The LLM guesser never appears in training.** Evaluation uses only LLM
guessers; every other guesser exists only for training. Training against
the LLM would fit the spymaster to how one specific model reads a clue
and would collapse the train/eval distinction entirely, leaving no
held-out listener to measure against. The LLM model id is part of
`EvalSuite.suite_id` for the same reason (`codenames/eval_suite.py`).

A consequence worth stating plainly: `scripts/pipeline/run_arena.py`'s
spymaster × guesser matrix is a training diagnostic, not a scoreboard.
Its numbers must never be presented as evaluation results.

## Reward values are a scoring-time knob

The four reward values (own / neutral / opponent / assassin, in
`codenames/game.py::ROLE_REWARD`) are a risk-aversion knob, not a fixed
property of the game. `expected_words` reads them per candidate word as
`c_w`, so changing the assassin's cost changes how conservative the
model is without touching anything else.

Two objectives are worth reporting rather than collapsing into one:
expected value across listeners, and worst-case or CVaR across them. The
reward values are what move a model along that curve, which only works
while they stay a parameter rather than a constant baked into a fitted
artifact.

## Composition lives in a config file, not in code

Which guessers are in the training pool
(`configs/guesser_pool.json`), which spymasters are baselines
(`configs/spymasters.json`), and which boards and listener make up the
frozen eval suite (`configs/eval_suite.json`) are all data, not code.
Registries know how to build things *from* a config and have no opinion
on what it should contain (`codenames/guessers/registry.py`,
`codenames/spymasters/registry.py`).

The reason is that every result is conditional on those choices, so they
have to be nameable: results are reported as "under configuration X, we
observe Y," never as unconditional truths. A composition buried in code
can't be cited in a results table, swept over, or diffed between runs.

## Method decisions

**Nothing here trains.** Every model is a scoring function written down
in closed form and evaluated against the frozen LLM suite. An earlier
direction — an MLP over a multi-space feature vector, trained on
simulated guesser rollouts — was built, measured, and removed; see
`docs/log.md` for what it was and what it scored.

If a future model does need fitting, the outcome of a (board, clue,
guesser) triple is directly simulable — full feedback on every action,
for free, unlimited times — so it is a supervised problem, not an RL
one. RL would deliver the same information through policy gradients over
a ~111k-action space, with high variance and no clean validation metric.
Two consequences of that earlier pass are worth not rediscovering: split
train/val by board seed rather than by row, since one board appears in
many examples and a row-wise split leaks it across the split; and keep
the reward values out of whatever gets fitted, per the section above.

**Multi-turn effects are the open direction.** Clue choice changes which
words remain, and a clue only has to be distinguishable from the clues
already given. Both argue for lookahead or a value function on top of a
working single-turn scorer, not instead of one.

**Optimization of any small, fixed parameter set** — `linear_scorer`'s
weights, `expected_words`'s `tau_gain`/`tau_pen` — should use CMA-ES,
Bayesian optimization, or grid search, not policy gradients. Neither has
been done; both sets are still their original illustrative constants.

**Linear scoring is a baseline, not a candidate.** A weighted sum over
roles composes to a single linear function, which cannot represent
threshold effects (0.75 to three words beats 0.45 to five, because
guessing is greedy and ranking is what matters) or the margin between
the weakest intended word and the strongest distractor.
`spymasters/linear_scorer.py` is kept to demonstrate that, not as a
serious contender.

## The clue vocabulary is an intersection

The legal clue vocabulary is the intersection of every built embedding
space's own vocabulary — currently GloVe, Numberbatch, and
Wikipedia2Vec, 111,440 words (`scripts/data/build_similarity_tensor.py`).
Every legal clue therefore has a real vector in every space, whether or
not the current model consults that space.

That matters for the cross-space safety extension left open in
`versions/expected_words.md`: a single-space model is only as safe as a
listener who shares its space, and checking a candidate clue's assassin
margin in spaces the model didn't select from is only possible because
the vocabulary guarantees the vector exists.

## Environment

- Targets an RTX 5080 (16GB, Blackwell/sm_120), which needs CUDA 12.8+ —
  pin the PyTorch build early rather than debugging this near a deadline.
- Data generation and any local embedding training are CPU-bound and
  parallel.
- Nothing here needs rented/cloud compute.
- **Memory design note:** the mmapped similarity tensor exists
  specifically to avoid holding embedding models resident across many
  worker processes. Python's copy-on-write does not protect large dicts
  across `fork` — refcount updates touch the pages and gradually copy
  them per worker. Keep word→index maps small or shared deliberately;
  the fp16 tensor array itself shares for free. See `codenames/arena.py`'s
  module docstring for a concrete RSS bug this caused and how it was
  fixed.

## References

- Koyyalagunta et al., "Playing Codenames with Language Graphs and Word
  Embeddings," JAIR 71 (2021). arXiv:2105.05885
- Stephenson, Sidji & Ronval, "Codenames as a Benchmark for Large Language
  Models" (2024). arXiv:2412.11373
- Archibald & Brosnahan, "Adapting to Teammates in a Cooperative Language
  Game" (2024). arXiv:2403.00823
- Bills, Archibald & Blaylock, "Improving Cooperation in Language Games
  with Bayesian Inference and the Cognitive Hierarchy," AAAI 2025.
  arXiv:2412.12409
- Speer, Chin & Havasi, "ConceptNet 5.5: An Open Multilingual Graph of
  General Knowledge," AAAI 2017
- Yamada et al., "Wikipedia2Vec," EMNLP 2020 (system demonstrations)
- Codenames AI Competition framework: github.com/stepmat/Codenames_GPT
