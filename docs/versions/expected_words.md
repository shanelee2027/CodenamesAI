# expected_words

Replaces `z_threshold` as the fifth baseline registered alongside
`random`/`centroid`/`linear_scorer`/`oracle` in `configs/spymasters.json`
with `"roles": ["baseline"]`. `z_threshold` itself is deleted (code, test,
config entry, registry entry, doc) rather than kept side by side.

## Why replace it, not tune it

The retired `z_threshold` baseline documented two structural problems
(its measurements are preserved in `docs/log.md`), both
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

Derived in full in [`../clue-selection-theory.tex`](../clue-selection-theory.tex)
(rendered: [`../clue-selection-theory.pdf`](../clue-selection-theory.pdf)),
which is the authoritative statement of this model. Summarised here only
far enough to say what the code does.

One assumption, one parameter: the guesser perceives board word `x` as
`zhat_x = z_x + eps_x` with `eps ~ N(0, sigma^2)` independent per word,
and selects unrevealed words in decreasing `zhat` order, stopping after
`k` selections or on the first word of `B`. `sigma` is "how far off the
guesser's read of any single word is," in per-clue z units.

With `A` our team's words and `B` every other board word, let

```
T = max_{w in B} zhat_w        # perceived score of the strongest distractor
W = argmax_{w in B} zhat_w     # which distractor that is
N = #{i in A : zhat_i > T}     # own words the guesser ranks above it
```

The guesser reveals `min(k, N)` words of `A`, and errs exactly when
`N < k` — and the word it then takes is `W`, no other. So the objective is
`G(k) - P(k)` for

```
G(k) = E[min(k, N)]            P(k) = E[c_W * 1{N < k}]
```

Everything reduces by conditioning on `T`: `T` is a maximum of
independents so `F_T` is a closed-form product of normal CDFs; given
`T = t` the `zhat_i` are unchanged and independent, so `N` is
Poisson-binomial with `p_i(t) = Phi((z_i - t)/sigma)`; and `c_W` is
conditionally independent of `1{N < k}` given `T`, so `P(k)` needs only
`cbar(t) = E[c_W | T = t]`, which falls out of the reversed hazard rate
`h_w = f/F` as `sum_w c_w h_w / sum_w h_w`. Both integrals are then single
integrals against `dF_T`, evaluated on a grid taking each cell's mass as
`F(hi) - F(lo)`.

**Notation drift, worth knowing.** The `.tex` calls the maximum `T` and
writes similarities as `z_i`/`z_w`;
`codenames/spymasters/expected_words.py` still calls it `D` and writes
`a_i`/`b_w`. Same quantities, and the code is correct — but the names
were changed in the paper and not in the code.

### Two things this gets right that the obvious formulations do not

**Conditioning on `T`, not multiplying marginals.** Comparisons against
one own word share that word's `eps`, and all own words face the same
distractor draws. Both dependencies are positive, so a naive product
*understates* survival: against Monte Carlo at `sigma=1.8` the product
gave 1.1206 expected words where the truth was 1.4939 — low by 0.37.

**`N` counts every own word above `T`, not the top `k` by our own
similarity.** The guesser has no idea which words we intended; it takes
own words as it meets them. Scoring `P(the top k all clear T)` is
strictly smaller — low by 0.02 expected words at `k=2` and up to 0.30 at
`k=4`, with the penalty correspondingly overstated. That error was in the
original spec and survived a Monte Carlo test, because the test simulated
the same wrong event: it confirmed the code matched its own model rather
than the game. The test now simulates `N`.

**`penalty(k)` charges the word the guesser would actually pick.**
Summing `c_w` over every distractor that might have broken through would
count several misses that cannot all happen.

**A linear `k - penalty` was tried and rejected.** It picked `k=4` on
every board at every width tested, because a linear gain term credits
every announced word +1 regardless of how endangered it is. `G(k)` is
sub-linear instead: its k-th increment is `P(N >= k) <= 1` and
non-increasing, so claiming more only helps when the words are clear
enough of the distractors to keep that probability near 1.

**Selection reduces to "each clue's own best k," and this is exact.** Per
clue the best `k` is unambiguous, so reducing every clue to
`(best_k, best_score)` before ranking clues against each other finds the
same joint maximum a literal scan over the full `(clue, k)` grid would.
What the model must *not* do — and doesn't — is fix `k` by a rule
independent of the score (as `z_threshold` did, counting own words above
a percentile bar) and only judge risk afterward.

### Verification

- Matches a 20M-sample Monte Carlo to 0.5 standard errors; the grid
  converges by 96 cells.
- After the `N` correction, agreement within ±0.002 expected words at
  every `k` on real boards (3M samples), against gaps of up to 0.30
  before.

## No thresholds, no fallback chain

`t`, `neutral_outside`, `opponent_outside`, `assassin_outside`, and
`guesser_noise_std` don't exist in this model, and neither does the
earlier `tau_gain`/`tau_pen` pair — two widths implied two different
guessers, and after conditioning on `T` the algebra is written in `sigma`
throughout, so no `tau` survives to name. Every `(clue, k)` pair gets a
finite score, so `argmax` always has an answer.

The only guard left is the one real degenerate case — a team with zero
unrevealed own words, which cannot occur in an actual game since a team
is never asked for a clue once it has none left — handled by returning
the most common legal clue at `number=1`, the same "never raise, never
return no clue" contract every baseline follows.

## Choosing `sigma`

`sigma = 2.5`, recorded explicitly in `configs/spymasters.json` rather
than left as a class default: this model is meant to be the fixed
opponent, so its settings are part of every future comparison's identity,
and editing a default would invalidate them with no config diff.

It was selected by the sweep in the `.tex`'s Results section: `sigma`
over three orders of magnitude, each value playing the same 100 positions
drawn from held-out board words, scored the way the game scores a turn.
`sigma=2.5` gave the best mean reward (1.314, s.e. 0.069) with zero
assassin hits and 80/100 clean finishes; the announced number falls
monotonically with `sigma`, from 3.96 at 0.25 to 1.13 at 10.0.

**The guesser in that sweep was Claude Sonnet — and the frozen eval
suite now uses Sonnet too.** It was originally a deliberate split from the
suite, which used Opus; the suite moved to Sonnet on cost (see
`docs/log.md`, "The eval guesser is Sonnet, not Opus"). So this model's one
parameter was chosen against the same listener it is evaluated on, which
flatters it slightly and is worth stating plainly rather than leaving for
someone to notice. Opus stays available for final headline numbers, where
the split is restored. See `docs/iteration-architecture.md`'s "two roles
for guessers".

**Stale class default.** `codenames/spymasters/expected_words.py` still
defaults to `sigma=1.8`, which the sweep shows is miscalibrated (mean `k`
2.67). Every real caller goes through the registry and gets 2.5 from the
config, and `scripts/tools/sweep_sigma.py` passes its own value, so no
pipeline path reads it — but a test constructing the class directly does.

## Superseded measurements

The 40-board announced-number table and the three-listener
`run_two_team_arena.py` smoke test that used to sit here were taken
against the original `tau_gain`/`tau_pen` formulation, before both the
conditioning fix and the `N` fix. They measured a model this one is two
revisions past, and have not been re-run. They are in `docs/log.md` with
their original commits.

What replaces them is the `.tex` sweep above, which is both current and a
better measurement — an LLM listener on held-out words rather than
synthetic single-space guessers.

### Inherited: the cross-space assassin problem

`z_threshold` measured a real, structural finding (see `docs/log.md`)
that this model inherits rather than fixes, since it is a consequence of
selecting on a single embedding space (`space="numberbatch"`), not of the
threshold mechanism that document was otherwise about: **a single-space
spymaster is only as safe as a listener that shares its space.** Nothing
in this model's z-margins constrains where the assassin sits in a space
it never consults.

The sweep above reports zero assassin hits at `sigma=2.5` over 100
positions against Claude Sonnet, which is encouraging but does not test
the cross-space case — an LLM listener is not a different embedding
space, it is a different kind of listener entirely.

### Inherited: embedding-space disagreement

Also carried over as context rather than re-derived: `z_threshold` traced
the cross-space assassin problem to a concrete example (seed 9, clue
"counter", assassin at z=-0.05 in numberbatch but +1.79/+1.77 in
glove/wikipedia2vec) — the same clue can look completely safe in the
space a spymaster looks at and simultaneously dangerous in one it
doesn't. `z_for_board`/`ClueStats` already carry mean/std for all three
spaces, so extending the penalty term (or adding a hard safety filter) to
require low `b_w` in every built space is a scoring change, not an
architectural one. Still open.

## Open for the next model

- The multi-space safety extension above (`ClueStats` already has the
  data).
- `sigma` was swept, but `max_rarity=10.0` and `space="numberbatch"` were
  not; how sensitive the model is to either is unmeasured.
- `penalty(k)` uses `P(N < k)` and `cbar(t)`, which charge the expected
  cost of the strongest distractor. A distractor that is dangerous
  specifically against a *stronger* intended word is not separately
  penalised for that. Whether it matters in practice is unmeasured.
- Rename `D` to `T` and `a`/`b` to `z_i`/`z_w` in the code, to match the
  paper.
