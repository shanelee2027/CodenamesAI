# CodenamesAI — Project Scope

Senior CS project. A Codenames clue-giving agent built around a learned clue
scorer over multiple word embedding spaces.

This document is the design spec. It is the source of truth for architecture
decisions. Read it before proposing changes; if a change contradicts something
here, say so explicitly rather than silently diverging.

---

## 1. What this project is

Codenames has two roles. The **codemaster** sees which board words belong to
which team and gives a one-word clue plus a number. The **guesser** sees only
the words and tries to pick their team's words from the clue.

This project builds the codemaster. The guesser is deliberately simple and
hand-written — it exists as a training environment, not as a deliverable.

**The motivating problem.** Standard word embeddings encode dictionary
semantics. They do not encode pop-culture or entity knowledge. Given a board
with KING, DREAM, CRAFT, and SWORD, a human might clue "Technoblade" (a
Minecraft YouTuber: Minecraft/craft, PvP/sword, the Dream SMP/dream, his
Potato King persona). GloVe has no vector for that word at all, and could not
produce the clue at any threshold.

**The approach.** Use several embedding spaces with different knowledge
profiles, and train a model to learn which space to trust for which clue.

### What is explicitly NOT this project

- **Not an LLM wrapper.** We do not prompt a language model for a clue and
  parse the response. The codemaster is a learned scoring function over
  numerical features. No LLM appears anywhere in the pipeline.
- **Not novel research.** Reusing published techniques is fine and expected.
  The contribution is a working, well-measured system.
- **Not a training-from-scratch embeddings project.** Three of four spaces are
  downloaded pretrained.

---

## 2. Architecture

### The three stages

Work happens at three distinct times. Keeping them separate is important.

**Build time (once).** Acquire four embedding spaces. Fix a clue vocabulary
(~250k legal single words) and a board vocabulary (~800 Codenames card words).
Precompute the full similarity tensor: `[n_clues × n_board_words × n_spaces]`,
fp16, memory-mapped, roughly 1.6GB. After this, embedding models are never
loaded again.

**Train time (repeatable).** Simulate millions of turns against a pool of
hand-written guessers. Each simulation yields a labeled example. Train an MLP
to predict turn outcomes from board+clue features.

**Play time (per turn).** Score every legal clue in the vocabulary with one
batched forward pass, compute expected reward at each possible clue number,
return the best `(clue, number)` pair.

### The embedding spaces

| Space | Source | Knowledge type |
|---|---|---|
| GloVe | Download (Stanford NLP) | Corpus co-occurrence, dictionary semantics |
| fastText | **Trained by us** on Fandom dumps | Pop culture, proper nouns, subword morphology |
| ConceptNet Numberbatch | Download (v19.08) | Commonsense typed relations |
| Wikipedia2Vec | Download (pretrained) | Encyclopedic entities, wiki link graph |

Only the fastText space is trained locally. The others are inputs, in the same
sense that ImageNet-pretrained features are inputs to a downstream vision model.

### The feature vector

Given a board state and a candidate clue:

1. Look up the clue's similarity to all 25 board words, in all 4 spaces.
2. Partition the 25 by role: own (9), opponent (8), neutral (7), assassin (1).
3. **Sort descending within each role group, per space.**
4. Pad revealed/missing slots with a sentinel (−1) and emit a validity mask.
5. Concatenate: 25 values × 4 spaces = 100, plus mask, plus scalars
   (own words remaining, turn index, score differential). ~115 dimensions.

**Why sorting matters.** Board word order carries no information, so the
representation must be permutation-invariant — otherwise the model wastes
capacity learning that position 3 and position 7 mean the same thing. Sorting
also makes rank position meaningful: "highest own-word similarity is 0.8" and
"ninth own-word similarity is 0.8" are entirely different situations.

**Why concatenate spaces rather than average them.** Averaging destroys the
signal we care about. "High fastText similarity, near-zero GloVe similarity"
identifies a domain-specific clue — exactly the Technoblade case. Averaged into
one number, that is indistinguishable from a uniformly mediocre clue.

**The model never sees words.** Only numbers. All linguistic knowledge lives in
the similarity tensor; the network learns to read a similarity profile.

### The model

MLP. Input ~115 → hidden (256, 256, 128) → 5 logits → softmax.

Output is a **distribution over k**, the number of own-words the guesser will
reveal before stopping (k ∈ 0..4). Not a scalar score.

Three reasons for a distribution:
- Clue number selection falls out arithmetically (below).
- Calibration is measurable — reliability diagrams are a real diagnostic.
- Risk aversion becomes a runtime knob, adjustable without retraining.

### Play-time scoring

```
E[reward | clue, n] = Σ_k P(k | clue) · reward(k, n)
best_n              = argmax_n E[reward | clue, n]
score(clue)         = E[reward | clue, best_n]
```

Reward: +1 per own word, 0 and stop on neutral, −1 and stop on opponent,
−10 and stop on assassin. The assassin penalty is the risk-aversion parameter.

**Score all ~250k candidates.** Do not build a candidate generator. With the
precomputed tensor, scoring the full vocabulary is one gather plus one small
forward pass — milliseconds on the target GPU. A generator would add a
component to debug and introduce recall loss for no benefit.

---

## 3. The guesser pool

### Its role

The guesser pool defines the training signal. "Is this clue good?" reduces
entirely to "will the guesser pick our words?" This makes the pool the most
consequential design decision in the project.

### Diversity must be in knowledge, not noise

A pool of one guesser plus Gaussian noise is **wrong** and will actively defeat
the project's goal. Consider: a GloVe-based guesser has no vector for
"Technoblade," so it guesses badly, so the label is negative, so the scorer
learns Technoblade is a bad clue. The system would train itself out of the
exact behavior we set out to produce. Noise does not fix this — humans do not
differ from GloVe by an epsilon, they differ by having different knowledge.

Target ~8 guesser types differing structurally:
- One per embedding space (GloVe, fastText, Numberbatch, Wikipedia2Vec)
- One or two blending several spaces
- One rank-based rather than score-based
- One with Gaussian noise on similarities
- One with a confidence threshold that stops early

**Hold two types out entirely.** Training code must never touch them. They are
the evaluation set.

### The pool is an unvalidatable assumption — treat it as one

The pool's composition cannot be validated from inside the simulation, because
the simulation is defined by it. Any result is conditional on a made-up
distribution. Three mitigations, all required:

1. **Make it explicit.** Pool composition lives in a config file, not in code.
   Results are reported as "under pool configuration X, we observe Y."

2. **Sensitivity analysis.** Sweep pool compositions (GloVe-heavy,
   fastText-heavy, uniform, adversarial), retrain, and report whether
   conclusions hold. Stable rankings mean robustness. Unstable rankings are
   themselves a finding: clue quality is teammate-relative.

3. **Fit the mixture to human data.** From logged human games, fit the mixture
   weights over the pool by maximum likelihood — a handful of parameters, so
   30–50 games suffices. This does not learn a human model from scratch; it
   locates humans within a model class we already built.

### The expected-value / robustness tradeoff

Two objectives, both worth reporting:

- **Expected value** across the pool. Produces brilliant high-variance clues.
  Technoblade survives.
- **Worst-case or CVaR** across the pool. Produces safe clues that work for
  most guessers. Technoblade dies.

The risk-aversion parameter moves along this curve. Report the tradeoff rather
than picking one — it is the more interesting result.

---

## 4. Method decisions and their justifications

Recorded so they are not silently revisited.

**Supervised learning, not RL.** Given a board, clue, and guesser, the outcome
is directly simulable — full feedback on every action, for free, unlimited
times. That is the condition under which the problem reduces to supervised
regression. RL would deliver the same information through policy gradients over
a 250k action space, with high variance, no clean validation metric, and
failure modes that take days to diagnose. Supervised training runs in minutes,
which enables twenty experiments a day.

If multi-turn effects are pursued later (clue choice changes which words remain),
add one-step lookahead or a value function **on top of a working supervised
scorer**, not instead of it.

**Nonlinear scoring, not tuned constants.** A weighted-average-of-spaces
followed by a weighted-sum-over-roles composes to a single linear function.
It cannot represent threshold effects (0.75 to three words beats 0.45 to five,
because guessing is greedy and ranking is what matters), margins (the gap
between our lowest target and their highest word), or space-conditional trust.
The 8-constant version is still built — as a baseline (§6).

**Optimization of any small parameter set** uses CMA-ES, Bayesian optimization,
or grid search. Not policy gradients.

**Split by board seed, not by row.** The same board appears in many training
examples. Row-wise splits leak boards across train/val and inflate validation
numbers.

---

## 5. Milestones

### Phase 0 — Start immediately, runs in parallel

**M0: Corpus collection.** Fandom database dumps have external latency —
requests are queued and processed during off-peak hours, sometimes taking up to
a week. Begin week one regardless of what else is in progress.

- Dumps are at the bottom of each wiki's `Special:Statistics` page.
- Choose **"Current pages"**, not "Current pages and history." History would
  multiply corpus size with near-duplicate text, which degrades embeddings.
- Files are `.7z` — install `p7zip-full`.
- Target 20–40 mid-size wikis across gaming, anime, film, TV. Breadth beats
  depth; Codenames boards touch everything.
- Target 1–5GB of extracted text. Below ~500MB gives noisy vectors.
- Note: some wikis (including Minecraft) have migrated off Fandom to
  independent MediaWiki hosts. The same `Special:Statistics` export usually
  exists there.
- Parse with `wikiextractor` (plaintext) or `mwparserfromhell` (if the link
  graph is needed). Do not hand-roll a wikitext parser.

### Phase 1 — Embeddings and the inspector

The goal of this phase is a tool that answers: *given this board and this clue,
what does each embedding space think, and how does a baseline scorer rate it?*

**M1: Board and legality.**
- `Board` class: 25 words, role assignment, revealed tracking, seeded
  deterministic generation.
- `is_legal_clue()`: rejects board words, substrings in either direction, and
  morphological variants.
- pytest coverage of legality edge cases: plurals both directions, hyphenated
  board words, multi-word entries, case sensitivity.

Bugs here silently inflate every downstream score.

**M2: GloVe and the similarity tensor.**
- Download script with resume and checksum verification. GloVe 6B-300d first
  (fast iteration); test 840B later.
- Clue vocabulary filter → ~250k legal lowercase alphabetic tokens above a
  frequency threshold (threshold is a CLI arg).
- Build the memory-mapped fp16 tensor plus a JSON word→index map. Normalize
  vectors first; compute on GPU in batches. Report peak VRAM and wall time.
- `similarity.py`: mmap loader exposing per-board similarity lookups.
- **`scripts/sanity_check_sims.py`** — prints top-20 nearest clues for sample
  board words. Do not proceed without eyeballing this output. If the tensor
  indexing is wrong, everything downstream is wrong and it will not be obvious.

**M3: The inspector.**
- CLI (and optionally a small web UI) taking a board and a typed clue, printing
  per-space similarity to all 25 words, per-space top-ranked words, what each
  guesser would pick, and a baseline score.
- This tool remains useful for the entire project. Build it well.

**M4: The remaining embedding spaces.**
- Download Numberbatch (v19.08) and Wikipedia2Vec pretrained.
- `scripts/train_fandom_fasttext.py`: extract and clean Fandom dumps, train
  fastText with gensim, export word vectors only (not the full `.bin` — the
  subword bucket table doubles the footprint for no post-training benefit).
  Subword units matter here; fan wikis are dense with rare proper nouns.
- Extend the tensor to 4 spaces. Extend the inspector.
- **Milestone test:** does the inspector show high fastText similarity between
  "technoblade" and KING/DREAM/CRAFT/SWORD, and near-zero GloVe similarity?
  This is the project's motivating case.

### Phase 2 — Simulation and learning

**M5: Guesser pool.** ~8 structurally different guessers per §3. Registry so
the arena can enumerate them. Two marked held-out; training code cannot access
them. Pool composition in config.

**M6: Arena.** Cross-play matrix — every codemaster × every guesser over fixed
seeded boards. SQLite logging, one row per turn: board seed, agent ids, clue,
guesses, outcome. Multiprocessing across cores; verify workers share the mmapped
tensor rather than each loading a copy (report per-worker RSS). Metrics: win
rate, mean turns, assassin rate, mean own-words per clue.

**Off-diagonal results are the ones that matter.** A codemaster that dominates
its training pool and collapses against held-out guessers has overfit to its own
guesser — the failure mode that makes agents useless to humans.

**M7: Features and data generation.**
- `features.py` per §2. Tests for permutation invariance and masking **before**
  anything is built on top of it. A bug here is silent and poisons everything.
- `scripts/generate_training_data.py`. Sample board (including partially
  revealed mid-game states), clue, number, and guesser; simulate; record
  `(features, k, reward)`.
- Clue sampling mix: ~60% top-k neighbors of board word subsets, ~30% near any
  board word regardless of role (this is where dangerous assassin-pulling clues
  come from — the model must see them), ~10% random.
- Target 5–20M examples. Appendable mmapped output. Report examples/second.

**M8: The scorer.** MLP per §2. Training script with board-seed splitting,
early stopping, checkpointing. Log training curves and validation reliability
diagrams. `codemasters/learned.py` implementing play-time scoring with a
runtime risk-aversion parameter. Register with the arena.

**M9: Evaluation and ablations.**
- Ablate each embedding space in turn.
- Ablate sorting (feed unsorted similarities).
- Ablate concatenation (average spaces instead).
- Pool sensitivity sweep per §3.
- Linear model over the same features, with coefficients reported — an
  interpretability check showing which spaces and rank positions carry weight.

**M10: Human evaluation.** Small web app; logged games with real people. Fit
the guesser mixture to that data. Compare human win rates against simulated
win rates — the gap is the most informative number in the project.

### Optional stretch

- **Custom retrofitting.** Retrofit the Fandom fastText vectors against a graph
  built from Fandom's internal link structure (page A links to B → edge).
  Faruqui et al. 2015: minimize `Σα‖qᵢ−q̂ᵢ‖² + Σ_(i,j)∈E β‖qᵢ−qⱼ‖²`. Convex,
  a few iterations of averaging, seconds to run. Yields a fifth space that
  exists nowhere else, plus an ablation: does graph structure help beyond the
  corpus alone?
- **Teammate conditioning.** Encode hint history — last k rounds of (clue,
  number, actual picks) — into a latent teammate vector concatenated to the
  features. Requires a heterogeneous pool so the model must infer which guesser
  it faces. Evaluate turn-1 (no history) vs turn-3+ (history) on held-out
  guessers; the gap should widen over a game.
- **Multi-turn lookahead** per §4.

---

## 6. Baselines

Every one of these must be implemented and beaten in order. If the MLP does not
clearly beat the linear scorer, something is wrong — diagnose before adding
anything.

1. Random legal clue
2. Centroid — clue nearest the mean of a random own-word subset
3. **Linear scorer** — the 8-constant design: weighted average across spaces,
   then weighted sum across roles (own +1, opponent −1, neutral −0.3, assassin
   −10), constants tuned by CMA-ES or grid search
4. Linear model over the full sorted/concatenated feature vector
5. MLP

Baselines 3 and 4 are the informative pair: 3 fails for the structural reasons
in §4, and the gap between 3 and 5 is the project's headline result.

---

## 7. Environment

- **GPU:** RTX 5080, 16GB, Blackwell (sm_120) — requires CUDA 12.8+. Pin the
  PyTorch build early rather than debugging this near a deadline.
- **CPU:** Ryzen 7 9800X3D, 8c/16t. Data generation and fastText training are
  CPU-bound and parallel.
- **RAM:** 64GB.
- **Cloud:** not required. Nothing here needs rented compute.

**Memory design note.** The mmapped similarity tensor exists specifically to
avoid holding four embedding models resident across 16 worker processes. Note
that Python's copy-on-write does not protect large dicts across `fork` —
refcount updates touch the pages and gradually copy them per worker. Keep
word→index maps small or shared deliberately; the fp16 array itself shares for
free.

---

## 8. Conventions

- Python 3.11+, type hints on public functions, pytest, no notebooks in VCS.
- `.gitignore`: `data/`, `cache/`, `__pycache__`, `.venv`, `*.npy`, `*.bin`,
  `*.db`.
- Commit at every milestone. Branch per experiment — arena results need to be
  comparable across versions.
- Keep `docs/log.md` updated as work proceeds, recording what was expected and
  what actually happened. Reconstructing six months of reasoning at writeup time
  is much harder than taking notes.

### Directory layout

```
data/          raw dumps and embeddings (gitignored)
cache/         similarity tensor, generated datasets, checkpoints (gitignored)
codenames/
  board.py         board state, roles, legality
  similarity.py    mmapped tensor loader
  features.py      board + clue -> feature vector
  guessers/        the guesser pool
  codemasters/     baselines + learned scorer
  scorer.py        the model
  game.py          single-team game loop
  arena.py         cross-play evaluation
scripts/
tests/
docs/
```

### Working with this codebase

- One module at a time. Do not build several at once on a fresh codebase.
- Before implementing anything non-obvious, describe two structural options and
  what breaks with each, then recommend one.
- Flag design choices the author might not notice. This is a graded project
  that will be defended orally; every file needs to be explicable by the author.

---

## 9. Current status

- [ ] M0 — Corpus collection (start immediately)
- [x] M1 — Board and legality
- [ ] M2 — GloVe and similarity tensor
- [ ] M3 — Inspector
- [ ] M4 — Remaining embedding spaces + fastText training
- [ ] M5 — Guesser pool
- [ ] M6 — Arena
- [ ] M7 — Features and data generation
- [ ] M8 — Scorer
- [ ] M9 — Evaluation and ablations
- [ ] M10 — Human evaluation

---

## 10. References

- Koyyalagunta et al., "Playing Codenames with Language Graphs and Word
  Embeddings," JAIR 71 (2021). arXiv:2105.05885
- Stephenson, Sidji & Ronval, "Codenames as a Benchmark for Large Language
  Models" (2024). arXiv:2412.11373
- "Improving Cooperation in Language Games with Bayesian Inference and the
  Cognitive Hierarchy" (2024). arXiv:2412.12409
- Speer, Chin & Havasi, "ConceptNet 5.5: An Open Multilingual Graph of General
  Knowledge," AAAI 2017
- Faruqui et al., "Retrofitting Word Vectors to Semantic Lexicons," NAACL 2015
- Yamada et al., "Wikipedia2Vec," EMNLP 2020 (system demonstrations)
- Codenames AI Competition framework: github.com/stepmat/Codenames_GPT
