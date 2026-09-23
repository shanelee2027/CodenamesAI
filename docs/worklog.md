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

### 3c. Local LLM logprobs as the teacher — ⬜ not started
- ✅ Checked DeepInfra: no usable logprobs. gpt-oss and Qwen2.5-72B return
  none, and Llama-3.1-8B returns only the chosen token's.
- ⬜ Run Qwen3-8B (FP8) locally on the RTX 5080 and score whole words to get a
  full probability over the board. That would be a soft-label teacher, or
  possibly a listener used directly.
- ⬜ First, quantify how much worse Qwen3-8B is than gpt-oss (and Sonnet) on
  fixed positions (see #4).

## 4. Evaluation and benchmarks

- ✅ Sonnet vs Opus as the guesser on 25 paired positions: no detectable
  difference, at about a fifth of the cost.
  - **Sonnet is now the evaluation guesser**; Opus is kept for final numbers
    only.
- ✅ gpt-oss as the cheap bulk guesser, 47× cheaper than Sonnet per call —
  **[in use]** for sweeps.
- ⬜ **Frozen benchmark `holdout_v1`**: 100 fixed boards built only from the
  150 held-out words, with a Sonnet guesser. It is designed but **has never
  been run**. Recent results use gpt-oss on ad-hoc seeds, so they are only
  comparable within one sweep.
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
