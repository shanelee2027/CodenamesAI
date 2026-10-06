# profile_temperature_listener

assoc_profile_listener with per-pick temperatures: the most accurate
single-score booster, made less confident about the guesser's later picks,
at single-score speed. Asked for as "the best GBT with the expected-reward
objective, but not pick index, which is slower".

## What changed

- At pick j the guesser picks from the words left by softmax(s / τ_j). Pick 1
  stays at τ = 1. τ for picks 2 / 3 / 4+ = 1.09 / 1.26 / 1.41, fitted for the
  assoc profile booster by maximum likelihood on gpt-oss val
  (`scripts/pipeline/train_sequential_listener.py --booster
  cache/listener_gbt_assoc_profile.txt --arm temperature`, saved to
  `cache/sequential_listener_assoc_profile_temperature.json`). They are
  almost the assoc booster's (1.09 / 1.25 / 1.42).
- The reward is pick_temperature_listener's exact one: expected own words
  minus role costs (+1 / −0.2 / −1 / −10), a guesser that never stops.
- Everything else is assoc_profile_listener's.

## Results

Sonnet 5.5 generated held-out set (scripts/tools/report_listener_accuracy.py):

| Listener | R² | pick 1 | 2 | 3 | 4 | vs incumbent [95% CI] |
|---|---|---|---|---|---|---|
| incumbent | 0.321 | 0.549 | 0.201 | 0.139 | 0.058 | — |
| assoc, frozen | 0.329 | 0.558 | 0.210 | 0.147 | 0.068 | +0.009 [+0.005, +0.013] |
| assoc + per-pick temperatures | 0.335 | 0.558 | 0.216 | 0.156 | 0.090 | +0.015 [+0.010, +0.019] |
| assoc, pick index (depth 9) | 0.337 | 0.557 | 0.220 | 0.163 | 0.093 | +0.017 [+0.012, +0.022] |
| assoc profile, frozen | 0.331 | 0.560 | 0.212 | 0.149 | 0.064 | +0.010 [+0.006, +0.015] |
| **assoc profile + per-pick temperatures** | **0.337** | 0.560 | 0.219 | 0.159 | 0.089 | +0.017 [+0.012, +0.021] |

On 30 opening boards: 2 on 13, 3 on 12, 4 on 5, mean 2.73 (assoc_profile
3.10, the incumbent 2.93), 1.0 s per clue, the same as assoc_profile. On the
play server and in the demo bundle as "Association-profile listener + pick
temperatures". No suite run yet.
