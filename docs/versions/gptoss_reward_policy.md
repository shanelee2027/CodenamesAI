# gptoss_reward_policy

`imitation_policy` fine-tuned by REINFORCE on the per-turn reward gpt-oss's
guesses earn on the policy's own clues.

**Training.** `scripts/pipeline/train_reward_policy.py`.
- **Reward.** gpt-oss-120b at temperature 1 ranks the unrevealed words with
  no number announced. The ranking scores the turn under the game's reward
  values (+1 own, -0.2 neutral, -1 opponent, -10 assassin) for every k at
  once.
- **Update.** REINFORCE with a learned baseline and a KL penalty to the
  imitation policy. The outcome head is trained on the observed outcome.
- **Positions.** Training vocabulary only, with seeds disjoint from
  imitation and validation.

**Result** (docs/log.md). Flat.
- Validation reward went 1.762 at the start, then 1.775, 1.759, 1.757, 1.769
  and 1.788 over 275 iterations. Every paired difference was inside +-0.03.
- The policy moved (KL 0.2-0.3; a fifth of its greedy clues changed), but
  its reward did not.
- One gpt-oss sample per board is too noisy for this. Shane stopped the run
  (`cache/policy_gptoss_reward.pt`, iteration 275, capped at 4) to change the
  objective to winning games: `win_actor_critic`.
