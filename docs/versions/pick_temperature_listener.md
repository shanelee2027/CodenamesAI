# pick_temperature_listener

`learned_listener` with the listener's confidence set separately for each
pick. Registry entry `pick_temperature_listener` in `configs/spymasters.json`;
code in `codenames/spymasters/pick_temperature_listener.py`.

## What is different

The incumbent's reward scores a clue once and renormalises the same scores
after every pick (Plackett-Luce, `codenames/pl_reward.py`). Measured on
rankings no listener was fitted on, that is calibrated at pick 1 and
increasingly overconfident after it. On held-out-word boards the favourite at
pick 3 is predicted 53% and picked 34% (docs/log.md, "Is the listener as sure
of its later picks as it should be?").

Here pick j is softmax(score / tau_j) over the words left.
- **Temperatures:** tau = 1.0, 1.1, 1.3, 1.5, with pick 4's used for every
  later pick.
- **How they were fitted:** by maximum likelihood per pick, for
  `cache/listener_gbt.txt`, on the clean-holdout boards (seed 1,040,000 on).
  That listener was trained on both train and val, so those were the
  out-of-sample set (`scripts/tools/listener_step_calibration.py --booster
  cache/listener_gbt.txt --fit-on "new boards"`).
- **The reward** is computed exactly over which own words have been picked
  (2^n_own states). That replaces pl_reward's closed form, which needs fixed
  scores. At tau = 1 the two agree, and both match brute-force enumeration
  (`tests/test_pick_temperature_listener.py`).

Everything else is the incumbent's: the same listener, shortlist, costs and
search. The incumbent got a one-method hook (`_gain_and_penalty`) so this
model overrides only the reward, and its own behaviour is unchanged.

## Results

**Choices** (600 real game states from the incumbent's validation games,
`scripts/tools/compare_pick_temperature.py`):
- 32% of positions get a different clue or number.
- Mean number 2.37 -> 2.03.
- 4s drop from 95 to 26. Uncapped, 5s drop from 33 to 1.

**Games** (holdout_v1_gptoss, 97 boards both ways, gpt-oss guessing, against
learned_listener):

| | win% | boards won both ways | mean k | own words / clue | own% |
|---|---|---|---|---|---|
| pick_temperature_listener | 44.3% [0.38, 0.51] | 8 | 1.92 | 1.56 | 85.4% |
| learned_listener | 55.7% [0.49, 0.62] | 19 | 2.16 | 1.61 | 82.5% |

Sign test on the 27 decisive boards: p = 0.052. **It loses.** Its guesses are
more accurate (85% against 83% own), but it scores fewer words per turn and
the game is a race.

## Why, and what is open

- **Calibration was not the whole story.** The incumbent's role costs
  (neutral 0.2, opponent 1, assassin 10) were swept with the overconfident
  reward, and nothing beat them. The overconfidence acted as a bonus for
  aggression. A per-turn objective that ignores tempo is too cautious for a
  race, and the two errors partly cancelled. Correcting one without retuning
  the other loses games. A fair test re-sweeps the costs (the neutral cost
  first) with this reward.
- **One tau per pick cannot tell a real three-word clue from a one-word
  clue announced for 4.** It flattens both. A pick-by-pick listener that is
  told which words were already guessed could make that distinction (the
  deferred sequential-listener plan).
- **Not run on Sonnet (holdout_v1).**
