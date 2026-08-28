# Design decisions

Standing design rationale that doesn't belong in the README's overview and
isn't tied to one model version. Recorded so it isn't silently revisited.
See [`docs/versions/`](versions/) for what changed between model versions,
and [`docs/log.md`](log.md) for the chronological record of how any of this
was actually arrived at.

## What this project explicitly is not

- **Not an LLM wrapper.** No language model is prompted for a clue anywhere
  in the pipeline. The codemaster is a learned scoring function over
  numerical features.
- **Not novel research.** Reusing published techniques is fine and expected;
  the contribution is a working, well-measured system.
- **Not a training-from-scratch embeddings project.** The embedding spaces
  are downloaded pretrained (fastText, trained locally on a Fandom corpus,
  is the one exception — see the status note in the main README).

## The motivating problem

Standard word embeddings encode dictionary semantics, not pop-culture or
entity knowledge. Given a board with KING, DREAM, CRAFT, and SWORD, a human
might clue "Technoblade" (a Minecraft YouTuber — Minecraft/craft, PvP/sword,
the Dream SMP/dream, his Potato King persona). GloVe has no vector for that
word at all and could not produce the clue at any threshold. The approach
here is to use several embedding spaces with different knowledge profiles
and train a model to learn which space to trust for which clue — not to
solve Codenames with a single "best" embedding.

## Feature vector design

- **Sort descending within each role group, per space.** Board word order
  carries no information, so the representation must be permutation-
  invariant — otherwise the model wastes capacity learning that position 3
  and position 7 mean the same thing. Sorting also makes rank position
  meaningful: "highest own-word similarity is 0.8" and "ninth own-word
  similarity is 0.8" are entirely different situations.
- **Concatenate spaces, don't average them.** Averaging destroys the signal
  that matters most: "high fastText similarity, near-zero GloVe similarity"
  identifies a domain-specific clue (the Technoblade case). Averaged into
  one number, that's indistinguishable from a uniformly mediocre clue.
  `codenames/ablation.py::average_concatenation` exists specifically to
  demonstrate this empirically, not as a serious alternative.
- **The model never sees words, only numbers.** All linguistic knowledge
  lives in the similarity tensor; the network learns to read a similarity
  profile, nothing else.

## The guesser pool

The guesser pool defines the entire training signal — "is this clue good"
reduces to "will the guesser pick our words" — which makes its composition
the single most consequential design decision in the project.

**Diversity must be in knowledge, not noise.** A pool of one base guesser
plus several Gaussian-noise levels is wrong and actively defeats the
project's goal. A GloVe-only guesser has no vector for "Technoblade," so it
guesses badly regardless of noise, so the label is negative, so the scorer
learns Technoblade is a bad clue — the system would train itself out of the
exact behavior the project exists to produce. Humans don't differ from
GloVe by an epsilon; they differ by having different knowledge. Every
guesser in `configs/guesser_pool.json` wraps a *structurally different*
knowledge source (a different embedding space), and noise is layered on top
of that real diversity, never used as a substitute for it.

**The pool is an unvalidatable assumption — treat it as one.** Its
composition can't be validated from inside the simulation, because the
simulation is defined by it. Any result is conditional on a made-up
distribution. Mitigations: make the composition explicit in a config file,
not code (`configs/guesser_pool.json`); report results as "under pool
configuration X, we observe Y," never as unconditional truths; and treat a
noise-level sweep or pool-composition sweep (`scripts/run_ablation_study.py`)
as a sensitivity check, not just a hyperparameter search.

**The expected-value / robustness tradeoff.** Two objectives are both worth
reporting, not collapsed into one: expected value across the pool (produces
brilliant high-variance clues; "Technoblade" survives), versus worst-case or
CVaR across the pool (produces safe clues that work for most guessers;
"Technoblade" dies). The risk-aversion reward parameters
(`codenames/scorer.py::reward_matrix`'s four reward values) move along this
curve at *scoring* time, not training time — see the current model version
doc for exactly how.

## First-pass simplifications (still in effect)

Three deliberate divergences from the guesser-pool design above, adopted to
sidestep the "different embeddings know different things" problem rather
than solve it head-on in the first working version. Revisit once fastText
exists and/or before scaling past a first pass.

1. **Clue vocabulary is an intersection, not a union**, of every currently-
   built embedding space's own vocabulary (currently GloVe, Numberbatch,
   Wikipedia2Vec — 111,440 words). Every legal clue has a real vector in
   every space, so no guesser can fail on a clue purely from a vocabulary
   gap — the exact effect a diverse pool exists to average out, which
   matters more when the pool is small. See
   `scripts/build_similarity_tensor.py`.
2. **Guesser pool is 3 members, not ~8**: one per currently-built embedding
   space, each wrapped in Gaussian noise, equally weighted, all
   training-visible, none held out. Still "diversity in knowledge, not
   noise" above — three different embeddings is genuine knowledge
   diversity — just a smaller pool than an eventual full version would use.
3. **Generalization is checked via held-out board words, not held-out
   guessers.** `codenames/assets/board_words_holdout.txt` holds 60 of the
   400 board words out of training data generation entirely
   (`codenames/board.py::load_training_wordlist()`), so a later evaluation
   pass can build boards entirely from unseen words to check generalization
   to unseen board *content*. This is orthogonal to what held-out guessers
   test (generalizing to an unseen *listener*), not a replacement for it —
   adopted because holding out 2 of only 3 guessers would leave a single
   training-visible guesser, reproducing the single-guesser anti-pattern via
   the held-out mechanism itself. The held-out-word evaluation is built but
   has never actually been run against a trained model — a real gap, not
   yet closed.

## Method decisions

**Supervised learning, not RL.** Given a board, clue, and guesser, the
outcome is directly simulable — full feedback on every action, for free,
unlimited times. That's the condition under which the problem reduces to
supervised classification. RL would deliver the same information through
policy gradients over a ~111k-action space, with high variance, no clean
validation metric, and failure modes that take days to diagnose. Supervised
training runs in minutes.

If multi-turn effects are pursued (clue choice changes which words remain,
cross-turn clue memory — see the current model version doc's open
questions), add lookahead or a value function on top of a working
supervised scorer, not instead of it.

**Nonlinear scoring, not tuned constants.** A weighted average of spaces
followed by a weighted sum over roles composes to a single linear function.
It can't represent threshold effects (0.75 to three words beats 0.45 to
five, because guessing is greedy and ranking is what matters), margins (the
gap between our lowest target and their highest word), or space-conditional
trust. The linear version is still built, as a baseline — see the README's
baselines section.

**Optimization of any small, fixed parameter set** (e.g. tuning
`codemasters/linear_scorer.py`'s baseline weights) should use CMA-ES,
Bayesian optimization, or grid search — not policy gradients. Not yet done
for that baseline; its weights are still SCOPE's original illustrative
constants, untuned.

**Split by board seed, not by row.** The same board appears in many
training examples (a board is sampled once, several clues are drawn against
it). Row-wise train/val splits leak boards across the split and inflate
validation numbers. `scripts/train_scorer.py` splits by a hash of each
example's board seed instead.

## Environment

- Targets an RTX 5080 (16GB, Blackwell/sm_120), which needs CUDA 12.8+ — pin
  the PyTorch build early rather than debugging this near a deadline.
- Data generation and any local embedding training are CPU-bound and
  parallel (see `scripts/run_ablation_study.py`'s `ProcessPoolExecutor`
  usage).
- Nothing here needs rented/cloud compute.
- **Memory design note:** the mmapped similarity tensor exists specifically
  to avoid holding embedding models resident across many worker processes.
  Python's copy-on-write does not protect large dicts across `fork` —
  refcount updates touch the pages and gradually copy them per worker. Keep
  word→index maps small or shared deliberately; the fp16 tensor array
  itself shares for free. See `codenames/arena.py`'s module docstring for a
  concrete RSS bug this caused and how it was fixed.

## References

- Koyyalagunta et al., "Playing Codenames with Language Graphs and Word
  Embeddings," JAIR 71 (2021). arXiv:2105.05885
- Stephenson, Sidji & Ronval, "Codenames as a Benchmark for Large Language
  Models" (2024). arXiv:2412.11373
- "Improving Cooperation in Language Games with Bayesian Inference and the
  Cognitive Hierarchy" (2024). arXiv:2412.12409
- Speer, Chin & Havasi, "ConceptNet 5.5: An Open Multilingual Graph of
  General Knowledge," AAAI 2017
- Faruqui et al., "Retrofitting Word Vectors to Semantic Lexicons," NAACL
  2015
- Yamada et al., "Wikipedia2Vec," EMNLP 2020 (system demonstrations)
- Codenames AI Competition framework: github.com/stepmat/Codenames_GPT
