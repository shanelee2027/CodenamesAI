# gbt_win_value

The GBT spymaster (`fresh_split_listener`) scored by the probability of winning instead of the expected reward. This is Section 5 of the thesis, policy iteration done on the GBT instead of the Gaussian (`gaussian_win_value`).

## What changes

`fresh_split_listener` takes the clue and number with the highest expected reward (1 per own word; −0.2 / −1 / −10 for a neutral word, an opponent word and the assassin) under its model of the guesser: the GBT's scores, with a temperature per pick. This model keeps that guesser model and search, and takes the clue and number with the highest

    W(c, k) = sum over endings P(ending | c, k) * (our win probability after it).

The ending probabilities are the same exact computation over picked sets (`tempered_gain_and_penalty`), called with indicator costs; `codenames/win_value.py::win_probability` combines them with V(a, b), the probability that the side to move wins with a own words left against b. Same code as `gaussian_win_value` and `win_prob_listener`.

## V by policy iteration

`scripts/pipeline/gbt_policy_iteration.py`:

1. Round 0: `fresh_split_listener` plays itself; V_0 fitted to who won from each score.
2. Round t: this model with V_{t-1} plays itself; V_t from its games.

- **Guesser:** `sim:fresh_split` (codenames/guessers/listener_sample.py): the GBT's scores on the clue, each pick sampled from softmax(s / tau_j) over the words left. In these games the spymaster's guesser model is exactly true. No API calls.
- **Boards:** 1,000 boards from the 150 tuning words per round, both seatings (2,000 games), the same boards every round. Not the training words (the trees are sharper on them), not the test words.
- **Shortlist 1000** in self-play (~0.75 s a clue against ~10 s for the whole pool; 95% of turns pick the whole-pool best at 1000).
- **V fit:** logistic, logit V quadratic in (a, b) plus 1/a and 1/b (`build_win_value.py --logistic`). In the Gaussian runs, isotonic and shrunk tables flipped near-tied number choices between rounds; only the logistic gave smooth ones.

Results (run `1118`, games in cache/selfplay_games.db under `gbtpi-1118-r{t}`, V in cache/policy_iteration/gbt/):

| round | policy | mean number | own words per clue | own% of guesses | assassin (share of a side's games) | turns per game | change in V |
|---|---|---|---|---|---|---|---|
| 0 | expected reward | 1.91 | 1.63 | 88.2% | 4.4% | 8.6 | |
| 1 | W, V_0 | 2.69 | 1.85 | 79.8% | 12.2% | 6.9 | 0.070 |
| 2 | W, V_1 | 2.64 | 1.87 | 80.3% | 12.2% | 6.8 | 0.0070 |
| 3 | W, V_2 | 2.62 | 1.86 | 80.6% | 12.2% | 6.9 | 0.0077 |
| 4 | W, V_3 | 2.61 | 1.86 | 80.7% | 11.7% | 7.0 | 0.0079 |

Change in V = game-weighted mean |V_t − V_{t−1}|. Two bootstrap resamples (over boards) of one round's games differ by 0.0125–0.0148, so V stops changing after round 2 up to sampling noise (the round-to-round changes come out below the resampling difference partly because rounds share boards and the simulated guesser's draws). `value_path` is `cache/policy_iteration/gbt/V_round4.npz`. V_4(9, 8) = 0.48: the side that moves first wins about half its games.

**Number choices** (scripts in the session scratchpad, docs/log.md 2026-11-18): no score has a two-peaked distribution of the number in any round (the isotonic Gaussian had 7 of 60), and mean number never goes against the score (bigger behind, smaller ahead). The opening is 2-4 (round 4: 31/45/22% for 2/3/4), drifting slowly toward smaller numbers over the rounds (mean 3.27, 3.16, 3.05, 2.91) while V changes by ~0.008.

## Not measured yet

- Any game against an LLM guesser. Everything above is self-play against the simulated GBT guesser. A Sonnet match (against the Sonnet spymaster, Sonnet guessing, test-word boards, as for the GBT spymaster) is about $9 of Anthropic spend and needs approval.
- V reads only the score, not which words are left.
