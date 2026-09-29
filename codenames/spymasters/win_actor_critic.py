"""A clue policy trained to win games against the learned listener.

The network is the clue policy's (codenames/clue_policy.py): raw clue-word
evidence and roles in, pi(clue | board) over the incumbent's pool out. Its
number comes from a k head, pi(k | board, clue) over 1..own words left, with
no cap at 4. Play is one forward pass: the most probable legal clue and its
most probable k. No listener runs.

Training (scripts/pipeline/train_win_actor_critic.py, docs/log.md
"win_actor_critic: design") has two steps:
1. Imitate the uncapped incumbent's clue and number, its picks only.
2. Actor-critic on full games against the learned listener, gpt-oss-120b
   guessing for both sides. The only reward is the game's result. A critic
   V(board) = P(win), learned by TD(lambda) from those games and reading the
   board's actual words, scores what each of the policy's clues really did.
   The actor moves toward clues that raised the win probability more than
   its other proposals for the same board.

Nothing in training or play reads the learned listener's probabilities.
docs/versions/win_actor_critic.md.
"""

from __future__ import annotations

from codenames.spymasters.imitation_policy import ImitationPolicySpymaster


class WinActorCriticSpymaster(ImitationPolicySpymaster):
    MODEL = "win_policy.pt"
