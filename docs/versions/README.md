# Model docs

One doc per model, named for the model — see `CLAUDE.md`'s naming rule
(descriptive names, not version numbers).

- [`learned_listener`](learned_listener.md): the incumbent.
- [`expected_words`](expected_words.md): the Gaussian baseline, and the
  incumbent's first stage.
- [`association_listener`](association_listener.md): the incumbent retrained
  with free-association counts, plus a pass priced on the absolute level they
  give.
- [`imitation_policy`](imitation_policy.md): a clue policy (one forward pass,
  no listener) imitating the incumbent; the control for the next one.
- [`gptoss_reward_policy`](gptoss_reward_policy.md): that policy fine-tuned by
  REINFORCE on gpt-oss's per-turn reward. Flat; stopped.
- [`win_actor_critic`](win_actor_critic.md): a clue policy trained by
  actor-critic on full games against the incumbent, with win/loss as the only
  reward.

The incumbent has been evaluated on the frozen suite once, against a Sonnet
spymaster (docs/log.md): 58% of games, sign p = 0.017.
