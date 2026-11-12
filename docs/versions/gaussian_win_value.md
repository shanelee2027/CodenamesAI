# gaussian_win_value

The Gaussian spymaster of `expected_words`, scored by the probability of
winning instead of the expected reward. This is Section 4 of the thesis.

## What changes

`expected_words` takes the clue and number with the highest expected reward,
1 per own word and -0.2 / -1 / -10 for a neutral word, an opponent word and
the assassin. These costs are set by hand. This model keeps the same guesser
model (independent N(0, sigma^2) noise on each word's standardized
similarity) and the same integral, and instead takes the clue and number
with the highest

    W(c, k) = sum over endings P(ending | c, k) * (our win probability after it).

The endings are "j own words, then a neutral word / an opponent word / the
assassin" and "all k own words". Their probabilities come from
`gain_and_penalty` called with indicator costs (1 on the opponent words, then
1 on the assassin). The win probability after an ending comes from V(a, b),
the probability that the side to move wins with a own words left against b.
`codenames/win_value.py::win_probability` combines the two; it is the same
code `win_prob_listener` uses. The only change to `expected_words.py` is that
its per-(clue, k) score is now a method, `_score_matrix`, which this model
overrides.

## V by policy iteration

V depends on how both sides play. `scripts/pipeline/gaussian_policy_iteration.py`:

1. Round 0: `expected_words` at the fitted sigma = 2.02 plays itself; V_0 is
   the fraction of positions with score (a, b) from which the side to move won.
2. Round t: this model with V_{t-1} plays itself; V_t from its games.

- Every game uses the simulated guesser that follows the model exactly at
  sigma = 2.02 (`gaussian_numberbatch_s2.02` in configs/guesser_gaussian.json),
  so the games are free.
- 2,500 boards of the 800 online words per round, both seatings, the same
  boards every round. Games go to cache/selfplay_games.db, not llm_store.db.
- V is pure weighted isotonic regression (non-increasing in a, non-decreasing
  in b), with no prior: `build_win_value.py --isotonic`. A cell with no games
  takes the middle of the interval monotonicity allows. With 5,000 games per
  round, 3 to 7 of the 81 cells were empty, all in corners.

Round-to-round change in V (game-weighted mean absolute difference): 0.080,
0.018, 0.019, 0.020. Two bootstrap resamples of one round's games differ by
0.015 (sd 0.002), so V stops changing after round 2 up to sampling noise.
`value_path` points at `cache/policy_iteration/V_round4.npz`.

## Prefilter

The prefilter of `expected_words` (the top 1000 candidates by a quick margin,
k = 1 to 3) was checked again under W, since a W-maximizer behind in the game
may want larger numbers. It matched the full model at 300 of 300 positions,
with numbers from 1 to 5, at 0.063 s per clue against 1.25 s.

## Result against gpt-oss

Against `expected_words` at sigma = 1.5 (the best sigma of the round robin),
gpt-oss guessing, 300 boards of all 800 words, both seatings
(label `winvalue-vs-ew15-800-p1`):

| | win% | 95% CI over boards | boards won both | assassin% | mean k | own/clue |
|---|---|---|---|---|---|---|
| gaussian_win_value (sigma 2.02) | 53.8% | [50.5, 57.2] | 65 | 17.5% | 2.24 | 1.34 |
| expected_words (sigma 1.5) | 46.2% | | 42 | 12.8% | 2.24 | 1.38 |

Sign test on the 107 boards one side won twice: p = 0.033. For comparison,
`expected_words` at sigma 2 won 43.2% [39.8, 46.3] against sigma 1.5 in the
round robin, so the change of objective moves the model near the fitted sigma
from clearly worse than 1.5 to better.

The numbers it gives depend more on the score: when behind, a mean of 2.79
against 2.64 for sigma 1.5; when ahead 1.37 against 1.51; when level 1.66
against 1.90. It hits the assassin more often, which is the cost of the risks
it takes when behind.

## Open

- V reads only the score, not which words are left.
- The guesser of policy iteration is the model itself; gpt-oss is not, and
  the gain here is measured against gpt-oss.
