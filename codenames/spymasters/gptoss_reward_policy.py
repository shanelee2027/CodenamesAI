"""A clue policy fine-tuned on the reward gpt-oss's guesses actually earn.

Starts from `imitation_policy`'s weights and is trained by REINFORCE
(scripts/pipeline/train_reward_policy.py): on fresh training boards it samples
a clue from its own pi(clue | board), announces the number its outcome head
prices best, and gpt-oss-120b (temperature 1) ranks the unrevealed words. The
ranking scores the turn exactly as the arena would, for every number at once.
The sampled clue's log-probability moves with its advantage over a learned
baseline, a KL penalty holds the policy near the imitation policy so it cannot
drift into clues that exploit gpt-oss's quirks, and the outcome head is
trained on the category the ranking fell into.

The case for it over the incumbent's search: the listener was fitted to clues
someone else chose, and the search then seeks out clues where the listener is
overconfident. This policy learns from the guesser's response to its own
clues. Play is the same as the imitation policy's (codenames/spymasters/
imitation_policy.py): no listener, no expected_words, one forward pass.
docs/versions/gptoss_reward_policy.md.
"""

from __future__ import annotations

from codenames.spymasters.imitation_policy import ImitationPolicySpymaster


class GptossRewardPolicySpymaster(ImitationPolicySpymaster):
    MODEL = "policy_gptoss_reward.pt"
