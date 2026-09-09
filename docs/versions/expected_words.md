# expected_words

Replaces `z_threshold` as the fifth baseline registered alongside
`random`/`centroid`/`linear_scorer`/`oracle` in `configs/spymasters.json`
with `"roles": ["baseline"]`. `z_threshold` itself is deleted (code, test,
config entry, registry entry, doc) rather than kept side by side.

## Why replace it, not tune it

`docs/versions/z_threshold.md` documented two structural problems, both
inherent to the threshold design rather than fixable by re-sweeping its
parameters:

1. **Its expected-reward term was measurably inert.** The role thresholds
   already rejected every candidate with a risky nearby distractor, so
   among survivors the risk term had nothing left to discriminate on --
   `score = k - risk` collapsed to `score = k` in practice. Making the
   term load-bearing meant loosening the pre-filter, which is a design
   change, not a tuning one.
2. **The thresholds required a three-stage fallback chain** for "no clue
   clears every threshold" -- a case that, measured over 1,235 turns of
   real play, never fired at the shipped defaults, but still had to exist
   and be tested for correctness.

This model has neither problem by construction: one metric,
`score(clue, k) = gain(k) - penalty(k)`, is defined and finite for every
`(clue, k)` pair, so there is no threshold to clear and no "no valid
clue" state to fall back out of.

## The metric

For each candidate clue (`rarity_percentile <= max_rarity`, default
10.0, `space="numberbatch"`): let `a_1 >= a_2 >= ...` be its z-scores
against unrevealed own words, descending, and `b_w` its z-score against
each unrevealed non-own word `w`, with cost `c_w = abs(ROLE_REWARD[role(w)])`.

```
q_w(x, tau) = Phi((b_w - x) / tau)                 # torch.special.ndtr
s_j     = prod_w (1 - q_w(a_j, tau_gain))            for j = 1..k
gain(k) = sum_{j=1..k} prod_{i<=j} s_i
penalty(k) = sum_w c_w * q_w(a_k, tau_pen)
score(clue, k) = gain(k) - penalty(k)
```

`tau_gain=2.5`, `tau_pen=0.7`, both in per-clue z units (not converted
through each clue's own sigma, unlike `z_threshold`'s risk term, which
did convert through sigma to land in raw cosine-similarity space where
its assumed noise model lived). Selected clue and number are the
`argmax` over every `(clue, k)` pair jointly, `k` ranging over
`1..min(len(own), MAX_CLUE_NUMBER)`.

`s_j` depends only on `j`, never on the outer `k` it's being summed
into -- it's a survival probability, "how likely is the guesser to still
be on an own word by the time they reach the j-th one." `gain(k)` is
deliberately sub-linear in `k` because of this: the k-th word's marginal
contribution is `prod_{i<=k} s_i <= 1`, discounted by the chance of
actually reaching it. `penalty(k)` uses only `a_k`, the weakest of the
intended `k` words, asking how likely a guesser who correctly reached
that word would then drift onto each distractor.

**A linear `k - penalty` was tried and rejected.** It picked `k=4` on
every board at every `tau` tested, because a linear gain term credits
every announced word +1 regardless of how endangered it is -- exactly
the failure mode `z_threshold`'s inert risk term had, reintroduced at the
gain side this time. The sub-linear product is what makes `k` a real
choice instead of always maxing out at `MAX_CLUE_NUMBER`.

**Selection reduces to "each clue's own best k," and this is exact, not
an approximation.** Since `s_j` doesn't depend on `k`, a fixed clue's
best `k` is unambiguous (whichever maximizes `gain(k) - penalty(k)`);
reducing every clue to `(best_k, best_score)` before ranking clues
against each other therefore finds the same joint maximum a literal
scan over the full `(clue, k)` grid would, without materializing that
grid as an explicit list. What the model must *not* do -- and doesn't --
is fix `k` by some rule independent of the score (as `z_threshold` did,
counting own words above a percentile bar) and only judge risk
afterward.

## No thresholds, no fallback chain

`t`, `neutral_outside`, `opponent_outside`, `assassin_outside`, and
`guesser_noise_std` don't exist in this model. The only guard left is
the one real degenerate case -- a team with zero unrevealed own words,
which cannot occur in an actual game since a team is never asked for a
clue once it has none left -- handled by returning the most common
legal clue at `number=1`, the same "never raise, never return no clue"
contract every baseline follows.

## Measured, 40 fresh boards (`load_holdout_wordlist()`, seeds 0-39)

| metric | measured | expected |
|---|---|---|
| mean announced number | 2.025 | ~2.0 |
| number distribution | {1: 1, 2: 37, 3: 2} out of 40 | concentrated on 2 |
| median assassin margin (`a_k - z_assassin`) | +4.32 | ~+4.3 |
| worst (minimum) assassin margin | +2.40 | ~+2.4 |

All three line up with the validated design's expected numbers. The
distribution being almost entirely `k=2` (37/40 boards), not spread
across 1-4, says the sub-linear gain term is doing real, board-dependent
work rather than defaulting to either extreme -- neither always-1 (which
a badly-miscalibrated risk term would produce) nor always-`MAX_CLUE_NUMBER`
(which the rejected linear variant produced on every board).

## Smoke test: `run_two_team_arena.py --n-boards 100`

Real two-team self-play, synthetic guessers (no API cost). Per this
model's own instructions, **these numbers are a smoke test only** --
does it play legally and finish games without catastrophe -- not an
evaluation result; these synthetic guessers rank by cosine in one space
and aren't the real evaluation target (the frozen LLM eval suite is).
Nothing here was tuned against them.

| listener | assassin-hit | half-turns (all) | mean clue number | correct/clue | own% |
|---|---|---|---|---|---|
| `noisy_numberbatch` (its own space) | 0.0% | 10.94 | 1.42 | 1.42 | 99.9% |
| `noisy_glove` | 14.0% | 11.06 | 1.46 | 1.25 | 87.2% |
| `noisy_wikipedia2vec` | 23.0% | 10.52 | 1.47 | 1.23 | 84.9% |

Every game finished legally across all three listeners; no crashes, no
stuck turns, no illegal clues. The clue number stays low (~1.4-1.5,
lower than `z_threshold`'s tuned 1.94-1.99) because this metric's
`gain(k)` genuinely discounts risk per word rather than counting words
above a fixed bar -- it isn't chasing a target `k`, it's directly
maximizing an expected-words-style objective, and with real per-board
distractors nearby, that objective often prefers announcing fewer,
safer words.

### Inherited: the cross-space assassin problem

`z_threshold.md` measured a real, structural finding that this model
inherits rather than fixes, since it's a consequence of selecting on a
single embedding space (`space="numberbatch"`), not of the threshold
mechanism that document was otherwise about: **a single-space spymaster
is only as safe as a listener that shares its space.** The assassin-hit
rate climbing from 0.0% (numberbatch listener) to 14.0% (glove) to 23.0%
(wikipedia2vec) above reproduces that same pattern -- nothing in this
model's z-margins constrains where the assassin sits in a space it never
consults. This is cited as inherited context from the prior model's
measurement, not as a new finding of this one; the smoke test above
wasn't designed to isolate it further; per `docs/design-decisions.md`'s
"diversity must be in knowledge, not noise" principle, this remains
expected behavior for a single-space model, not a bug.

### Inherited: embedding-space disagreement

Related, and also carried over as context rather than re-derived here:
`z_threshold.md` traced the cross-space assassin problem to a concrete
example (seed 9, clue "counter", assassin at z=-0.05 in numberbatch but
+1.79/+1.77 in glove/wikipedia2vec) -- the same clue can look completely
safe in the space a spymaster actually looks at and simultaneously
dangerous in a space it doesn't. This model's `z_for_board`/`ClueStats`
machinery carries mean/std for all three spaces already; extending the
*penalty* term (or a hard safety filter) to require low `b_w` in every
built space, not just the selected one, was left open by the prior model
and remains open here -- this model didn't attempt it, since the task
was to remove thresholds from the existing single-space design, not to
change what space(s) it consults.

## Open for the next model

- The multi-space safety extension noted above (`ClueStats` already has
  the data; this would be a scoring change, not an architectural one).
- Only `space="numberbatch"` and the shipped `tau_gain`/`tau_pen`
  defaults are exercised by the measurements above; a `tau` sweep
  (analogous to `z_threshold.md`'s `own_top`/`assassin_outside` sweep)
  hasn't been run, so how sensitive the announced-number distribution is
  to those two constants specifically is unmeasured.
- `penalty(k)` uses only the single weakest intended word, `a_k` -- a
  distractor closer to a *stronger* intended word than to the weakest
  one is not separately penalized for that. Whether this matters in
  practice (versus the sub-linear `gain` term already discouraging large
  `k` when any own word is weak) is unmeasured.
- Like `z_threshold`'s risk term, this model's guesser-uncertainty model
  is a single Gaussian-CDF-in-z-space stand-in for
  `codenames/scorer.py::expected_reward_and_best_n`'s actual learned
  `P(k, cause | clue)` -- a deliberate simplification, not a claim about
  how real guessers behave.
