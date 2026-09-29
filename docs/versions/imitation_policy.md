# imitation_policy

A clue policy that imitates the incumbent. It is the control for
`gptoss_reward_policy`, which starts from its weights.

**Model.** `codenames/clue_policy.py`: a DeepSets network over the 25 board
words.
- **Inputs.** For each (pool clue, board word): 25 raw evidence columns
  (five embedding z-scores, SWOW, PMI, WordNet, spelling and gloss
  overlaps), 12 word features and the word's role.
- **Outputs.**
  - pi(clue | board) over the incumbent's 10,674-clue pool, legal clues
    only.
  - An outcome head: how the turn ends, as j own words and then a miss on
    some role. Its expected reward per k picks the number.
- **At play time.** One forward pass: no listener, no expected_words.

**Training.** `scripts/pipeline/train_imitation_policy.py --head outcome`,
on 30k simulated training-vocabulary positions from
`scripts/data/collect_imitation_data.py`.
- Cross-entropy on the incumbent's clue.
- MSE between the outcome head's expected rewards and the incumbent's net
  values, over its 200-clue shortlist.
- Capped at k = 4 (`cache/policy_imitation.pt`, epoch 9).

**Results** (docs/log.md).
- Its greedy clue is the incumbent's 51.9% of the time on validation
  positions.
- Frozen suite (holdout_v1), Sonnet guessing: 42.5% of games against the
  incumbent's 57.5%, sign p = 0.036.
- It asks for more (k 2.34 vs 2.21), lands a smaller share (79.8% vs
  84.1%) and loses 20 games to the assassin against 12. The number is its
  weak point.
