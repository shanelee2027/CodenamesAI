# Worklog

A running checklist of everything tried and still to try, kept short enough to
walk someone through. Details and full numbers are in [`log.md`](log.md), under
the heading named in each entry.

**Legend**

- ✅ done · 🔄 in progress · ⬜ not started
- **[in use]** part of the current system · **[kept]** kept as a baseline or
  tool · **[built, unused]** built, but not in the deployed configuration ·
  **[rejected]** tried and dropped
- The **guesser** is named for every game result, because results against
  different guessers are not comparable: *Sonnet* is Claude Sonnet 5 at medium
  effort; *gpt-oss* is gpt-oss-120b at low effort, the cheap bulk listener.
- Per-turn stats are **k** (mean clue number), **own%** (share of guesses that
  hit an own word), **own/turn**, **reward/turn** (+1 own, −0.2 neutral,
  −1 opponent, −10 assassin), and **assassin%** (games lost to the assassin).

---

## 1. Centroid — **[kept]** as the reference opponent

- ✅ Chooses the clue nearest the mean vector of its own words. It has no
  penalty for being near the opponent's words, the neutrals or the assassin.
- Results come from matchups; there are **no self-play games with Sonnet**.
  Pooled over all 580 of its Sonnet games:
  - k 1.51, own% 81%, own/turn 1.15, reward/turn 0.78, assassin% 11.2%.
- Against `expected_words` at σ=2.5 (100 games, Sonnet): **49–51**, a coin
  flip. But centroid's assassin rate was 13% against 3% (p=0.009). In 7 of
  its 13 assassin losses, the assassin was the guesser's **first** pick.
  - log: "centroid vs expected_words, 100 games against Claude Sonnet"

## 2. Gaussian listener model (`expected_words`) — **[kept]** as the baseline, and first stage of #3

- ✅ Models the guesser as seeing similarity plus Gaussian noise, N(0, σ²),
  then chooses the clue and number that maximise expected reward. This is the
  model the paper is about: `clue-selection-theory.tex`.
- ✅ **Checked that cosine similarity isn't distributed the same for every
  clue word**, which is why the model works in per-clue z-scores
  (`clue_stats.npz`) rather than raw cosines (`scratch/hub_effect.ipynb`,
  2,000 sampled clues):
  - A clue's average similarity to the 400 board words varies a lot. In GloVe
    the middle 90% of clues span −0.10 to +0.06, about two standard
    deviations of a typical clue's spread (0.08).
  - So the same raw cosine means different things: 0.3 is z = 2.7 for one
    clue and z = 5.4 for another in GloVe (1.5 to 3.4 in Wikipedia2Vec).
  - The spread differs by clue too (GloVe standard deviation 0.064 to 0.096
    across the middle 90% of clues), so the z-score needs each clue's own
    mean *and* spread.
- ✅ **σ sweep** (the assumed noise level):
  - ✅ Per-turn reward sweep (100 positions per σ, Sonnet): flat, with no
    significant optimum. It did not predict game results.
  - ✅ **Full games against centroid**, 100 each, Sonnet:

    | σ | win% | k | own% | own/turn | reward/turn | assassin% |
    |---|---|---|---|---|---|---|
    | 1.0 | 69 | 2.67 | 72% | 1.48 | 0.85 | 14% |
    | 1.25 | 70 | 2.27 | 75% | 1.43 | 1.01 | 8% |
    | **1.5** | **70** | 1.93 | 79% | 1.33 | 1.01 | 6% |
    | 2.0 | 57 | 1.42 | 88% | 1.17 | 1.05 | 2% |
    | 2.5 | 51 | 1.16 | 93% | 1.07 | 0.98 | 3% |

    - **σ=1.5 chosen**: the win rate plateaus from 1.0 to 1.5, and 1.5 has the
      fewest assassin losses on that plateau. The lesson: tempo wins races.
    - ⬜ `configs/spymasters.json` still says σ=2.5. Changing it needs its own
      `docs/versions/` entry.
  - ✅ **Descriptive σ**, fitted to cached rankings at no cost: Sonnet 2.06,
    gpt-oss 2.20. The σ that best *describes* the guesser is not the σ that
    *plays* best.
- ⬜ Idea: exclude or penalise clues already given this game. The model
  re-gives a clue right after it failed (8.6% of clues at σ=1.5).
- ⬜ Idea: let σ grow with k (the fitted σ rises from 1.97 at k=1 to 2.63 at
  k=4).

## 3. Distilled listener: a GBT trained on LLM rankings (`learned_listener`) — **[in use]**, the incumbent

- ✅ A LightGBM model learns to predict how gpt-oss ranks the board for a clue,
  as a Plackett–Luce model. The spymaster shortlists 200 clues with the
  Gaussian model at σ=1.5, then scores that shortlist with the GBT's exact
  expected reward (`pl_reward.py`, derived in `clue-selection-learned.tex`).
- ✅ **Listener quality** on clean held-out boards, measured by McFadden R²
  (how much better than chance it predicts the guesser):
  - GBT **0.346**, against **0.226** for the Gaussian.
  - That is 53% more of the available information.
- ✅ **Games** (Sonnet as guesser, so this is a transfer test; the GBT learned
  from gpt-oss):
  - vs centroid: 82% (40 games).
  - vs σ=2.5: 90% (40 games).
  - vs **σ=1.5: 72%** (100 games, CI [0.63, 0.80]).
  - Against σ=1.5: k 1.97, own% 87%, own/turn 1.65, reward/turn 1.50,
    assassin% 1%. That is the same boldness as σ=1.5 but more accurate.
- ✅ Teacher data: 25.7k gpt-oss rankings bought, about $3 in total.
  Accuracy plateaued: tripling the data changed nothing.
- ✅ The teacher is 84–99% self-consistent when re-asked, so the gap is
  knowledge our features lack, not noise.
- ✅ Feature blocks, in the order tried:
  - **Rejected:**
    - Tier-1 extra embedding features: redundant.
    - Cohesion: no effect.
    - Raw rival similarities: slightly harmful, reverted.
    - Cross-encoders: wrong objective.
  - **Kept:**

    | Block | Gain (nats) |
    |---|---|
    | SWOW word associations | largest early gain |
    | Reverse SWOW | +0.029 |
    | Language-model PMI | +0.019 |
    | GloVe-840B + fastText | +0.008 |
    | Concreteness + WordNet | small |

  - **Marginal:** entity vectors.
- ✅ Training tweaks:
  - **Adopted:** down-weighting later choices, learning rate 0.01, and
    averaging PMI across templates.
  - **No effect:** seed ensembling (and diverse ensembles) and a
    hyperparameter re-sweep.
- ✅ Independence check (does removing one word leave the others' odds
  unchanged?): it fails. The model's top word flips about 20% of the time.
  The spymaster re-scores every turn, so this only affects look-ahead
  within a turn.
- ✅ Acronyms removed from the clue pool (`phd`, `nasa`, `uk`). **[in use]**
- ✅ **Role-cost sweeps** (the penalties for neutral, opponent and assassin),
  gpt-oss, 100–300 boards per setting:
  - Nothing beats the incumbent's 0.2 / 1.0 / 10.
  - Every cost increase loses: opponent cost is monotone worse from 0.5 to
    3.0, and assassin 20 or 40 loses 37%.
  - A later confirmation run on the assassin axis (~560 games per setting)
    agrees: ass=7 52.9% [0.49, 0.57], ass=40 37.1%. It is not in `log.md`
    yet.
  - gpt-oss rewards tempo over caution.

### 3a. k=1 tiebreak — ✅ built · **[in use in the play app; off in the arena]**
- ✅ First version: at k=1, play the most-similar clue for the target word.
  - 40 games against itself (Sonnet): 55–45, not significant.
  - **[rejected]** as unsafe: it ignores the board, so BIRD for EAGLE hands
    over CHICK.
- ✅ Replacement: among clues within 0.1 expected reward of the best, pick the
  most obvious. It fires on 9 of 18 k=1 positions and costs at most 9.5% of a
  turn.
- ⬜ Never measured in games. It is on in the play server and off in the arena
  sweeps.

### 3b. Decoys, so the listener can "pass" — ✅ built · **[built, unused]**
- The problem: the GBT only ranks board words against each other, so it
  cannot tell a confident clue from a vague one.
- ✅ Asking gpt-oss to pass is **[rejected]**. Its pass rate has no relation to
  clue quality (r = −0.07).
- ✅ **Decoys**: off-board words mixed into the candidates. A decoy beating
  every board word is a real signal: within the shortlist, it costs 27 points
  of accuracy and multiplies the assassin rate by 7.6.
- ✅ Collected 5,039 decoy positions, with 2, 5 or 10 decoys each and the
  ranking cut at the first decoy. Retrained as `listener_gbt_decoy.txt`.
  - The fitted level is **~4.1 nats** below the best board word and does not
    depend on the decoy count. Board accuracy is unchanged (0.478 vs 0.477).
- ✅ **Outside option** in the reward (`outside_n`): treat "guesser picks a
  decoy" as ending the turn at cost 0, the same as passing.
  - Swept against gpt-oss, 2,820 games: **a monotone loss**.

    | outside_n | win% |
    |---|---|
    | 1 | 48 |
    | 5 | 48 |
    | 10 | 45 |
    | 25 | 36 |
    | 50 | 31 |

  - The mechanism works (k falls from 2.00 to 1.54 and assassin losses halve),
    but gpt-oss never stops guessing, so caution does not pay against it.
    **`outside_n` stays 0.**
  - The case for it now rests on human guessers, who do stop.

### 3c. Local LLM log-probs as the teacher (Qwen3-8B-FP8) — ✅ done, not adopted
- ✅ Checked DeepInfra: no usable logprobs. gpt-oss and Qwen2.5-72B return
  none, and Llama-3.1-8B returns only the chosen token's.
- ✅ Qwen3-8B-FP8 runs locally on the RTX 5080 (9.4 GB). It is given the same
  prompt and word order as gpt-oss, and each word's probability is that of
  the continuation `["<word>",` of its JSON answer. Every step of every
  position then has a full distribution rather than one sampled pick. About
  5 positions/s, free.
- ✅ Qwen's confidence against gpt-oss's first pick, 3,026 positions: when
  Qwen is 99%+ sure, gpt-oss picks the same word 75% of the time; at
  90–99%, 39%; overall 49% at an average confidence of 0.86. When they
  disagree, Qwen gave gpt-oss's pick a median probability of 0.002. So it is
  overconfident as a model of *another* guesser, which is why the raw
  distribution is not used directly as the listener.
- ✅ Qwen is the whole teacher. Every position gpt-oss was asked about is
  rescored by Qwen, with step 2+ conditioned on **Qwen's own** earlier
  picks, at T=1 with nothing fitted to any other guesser. gpt-oss
  contributes only which boards and clues were asked (which our sampler
  chose). 26k positions at ~6 positions/s, about 70 minutes.
- ✅ Trained two listeners on exactly the incumbent's positions and features:
  Qwen soft (full distributions) and Qwen hard (argmax only).
- ✅ Compared against observed picks, McFadden R^2 (pooled / step 1):

  | Model | Sonnet | gpt-oss holdout |
  |---|---|---|
  | incumbent (gpt-oss labels) | **0.553 / 0.735** | **0.342 / 0.553** |
  | Qwen soft | 0.476 / 0.653 | 0.229 / 0.426 |
  | Qwen hard | 0.476 / 0.654 | 0.226 / 0.427 |
  | Qwen raw | -0.262 / -0.016 | step 1: -0.582 |

  - The Qwen-taught listener is ~0.08 worse on Sonnet, the guesser neither
    teacher is. **Not adopted.**
  - Soft = hard: Qwen is so peaked that its distributions add nothing over
    its argmax.
  - Raw Qwen is worse than guessing uniformly (confidently wrong);
    distilling it through the features is what makes it usable.
- ⬜ Head-to-head games against the incumbent on the frozen suite: not
  worth the money unless something changes, since it lost on prediction.

### 3d. Does the way gpt-oss is asked change its picks? — ✅ done
- ✅ 300 holdout positions, 8 samples each at temperature 1. Four prompts:
  - the training prompt (clue and number, rank everything);
  - the same without the number;
  - one word at a time, with earlier picks removed;
  - one word at a time, also told "your first guess was correct".
- ✅ **Step 1 does not depend on the prompt.** Distributions differ by at
  most D = 0.04 (squared L2; disjoint = 2). The incumbent listener scores all
  three prompts' picks within 0.01 R^2 (0.603–0.613).
- ✅ **Step 2 does.** Continuing the list vs re-asking: D ≈ 0.12, about 5
  points of agreement below self-agreement. Being told the first guess was
  correct moves it again (D = 0.06).
  - The re-ask is closer to what our model assumes: it picks step 1's
    runner-up 62% of the time, against 51% for the list continuation.
  - The listener predicts both about equally (0.301 on the training prompt,
    0.305 on the re-ask, 0.278 when told the first guess was correct).
  - The arena's LLM guesser reads one ranking per turn, so for the arena the
    training prompt is the matching one.
- ✅ Found on the way: the training rankings were bought with no temperature
  set, and DeepInfra then answers an identical prompt identically in runs.
  The listener is fitted to gpt-oss's near-greedy picks, not its sampled
  distribution.

### 3e. Association counts, so the listener knows how good a clue is — ✅ built · **[built, not yet used in play]**
- The problem again: the board softmax cannot tell (4, 2, 2) from (2, 0, 0).
  Any target that fixes one level per clue solves it.
- ✅ Asked gpt-oss for 25 free associations of each of the 9,494 clues, with
  no board shown, 5 samples each (47k calls). A board word's target is how
  many of the 5 lists name it: an absolute rate, not normalised over the
  board.
- ✅ Added a Poisson term (generalized KL, the Bregman divergence for counts)
  on step-1 rows, alongside the board softmax. Slope 0.87, taken from the
  incumbent's own scores.
- ✅ Results on held-out boards. "Level ρ" is the Spearman correlation of
  predicted vs observed association total per board, i.e. the number the
  softmax cannot learn:

  | Model | board R^2 (val) | Sonnet R^2 | gpt-oss holdout R^2 | level ρ | level ρ, unseen clues |
  |---|---|---|---|---|---|
  | incumbent | **0.355** | 0.553 | **0.342** | 0.577 | 0.414 |
  | weight 0.3 | 0.351 | **0.554** | 0.337 | 0.710 | 0.513 |
  | weight 1 | 0.342 | 0.547 | 0.325 | 0.746 | 0.542 |
  | weight 3 | 0.327 | 0.535 | 0.308 | **0.765** | **0.556** |

  - The incumbent already knows part of the level (ρ 0.58), because some
    features are raw rather than board-relative.
  - At weight 0.3 the level improves a lot (+0.10 on unseen clues) for
    almost no board cost (Sonnet +0.001, gpt-oss holdout −0.005).
  - Higher weights trade board accuracy for level.
  - Decoy-board R^2 drops slightly (0.213 → 0.207). Decoys measure the
    level against random words under the same clue, and a clue-wide shift
    moves those words too, so they cannot see what this adds.
- ⬜ Next: use the level in play. Price "pass" as a fixed rate against
  exp(a·s + b) instead of a random word's score.

## 4. Evaluation and benchmarks

- ✅ Sonnet vs Opus as the guesser on 25 paired positions: no detectable
  difference, at about a fifth of the cost.
  - **Sonnet is now the evaluation guesser**; Opus is kept for final numbers
    only.
- ✅ gpt-oss as the cheap bulk guesser, 47× cheaper than Sonnet per call —
  **[in use]** for sweeps.
- 🔄 **Frozen benchmark `holdout_v1`**: 100 fixed boards built only from the
  150 held-out words, Sonnet guesser, played head to head in both seatings.
  Runnable (`scripts/pipeline/run_eval_suite.py`) but **not yet run**. Recent
  results use gpt-oss on ad-hoc seeds, so they are only comparable within one
  sweep.
  - ⬜ Pin the incumbent's config for it, including whether the k=1 tiebreak
    is on.
- ⬜ Teacher benchmark: fixed positions, scoring Qwen3-8B, gpt-oss and Sonnet
  on how often they agree and on how well each one guesses.
- ✅ **Blind human study** (`/eval`): one clue per position, spymaster hidden,
  stopping recorded. Analysis script included.
  - ⬜ Play the ~100-position pilot, then fix the sample size *before*
    collecting more.
- ✅ Local play app with a spymaster picker (incumbent, decoy and
  outside-option variants), plus a laptop demo bundle.
- ✅ Phone artifact.
  - ⬜ It still uses the old max-similarity k=1 rule, not the tiebreak.

## 5. Infrastructure (not experiments, but they took time)

- ✅ Response cache (`cache/llm_store.db`), so every paid LLM answer is
  reusable and past results are reproducible.
- ✅ Arena parallelism: 6 processes × 16 threads pulling from a shared queue,
  with the per-turn search memory-guarded. About 12 h → 84 min for the
  `outside_n` sweep.
