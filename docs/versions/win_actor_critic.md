# win_actor_critic

A clue policy trained to beat the learned listener in full games. The only
reward is the game's result. Design: docs/log.md, "win_actor_critic: design".

**What is different from the earlier policies.**
- **Objective.** Win probability against `learned_listener`, not per-turn
  reward. No hand-set values (+1 / -0.2 / -1 / -10) appear anywhere in
  training.
- **Nothing reads the learned listener's probabilities.** The listener is
  only the opponent, and its picks (never its values) are the imitation
  warm start. There is no learned guesser model either, which was Shane's
  rule.
- **The number.** A k head, pi(k | board, clue) over 1..own words left: no
  cap at 4.

**Pieces.**
- **Actor.** The clue policy network (`codenames/clue_policy.py`,
  `head="k"`). It is warm-started by imitating the uncapped incumbent's
  (clue, k) picks, saved as `cache/win_policy_init.pt`.
- **Critic.** `codenames/win_critic.py`: V(board, side to move) = P(mover
  wins), by attention over the unrevealed words.
  - Pair features keep magnitudes, margins, crowding and each similarity
    source separately.
  - It is learned by TD(lambda) from real games: `cache/win_critic.pt`, then
    the copy updated during training.
- **Games.** `codenames/win_game.py`, with the same rules as
  `play_two_team_game` (tested). gpt-oss-120b at temperature 1 guesses for
  both sides. On the agent's turns it ranks with no number, so one real
  ranking fixes the board after every k.
- **Update.** `scripts/pipeline/train_win_actor_critic.py train`.
  - For each proposal, Q(k) = P(win) after the first k words of its real
    ranking.
  - The clue term is REINFORCE with a leave-one-out baseline across the
    branched proposals for the same board.
  - The k term takes the exact expectation over k.
  - A KL penalty keeps the policy near the imitation start.

**Results.** See docs/log.md, from "win_actor_critic: critic pilot" on.
