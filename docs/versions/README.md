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
- [`pick_temperature_listener`](pick_temperature_listener.md): the incumbent
  with per-pick listener temperatures in its reward.
- [`isa_listener`](isa_listener.md): the incumbent with a listener that has
  directional WordNet is-a features.
- [`conceptnet_listener`](conceptnet_listener.md): isa_listener plus ConceptNet
  relations and phrases.
- [`assoc_feature_listener`](assoc_feature_listener.md): conceptnet_listener
  plus gpt-oss's free associations as listener features.
- [`win_prob_listener`](win_prob_listener.md): the probability of winning as the
  objective, with any listener booster.
- [`within_turn_listener`](within_turn_listener.md): conceptnet_listener with
  later picks of a turn modelled separately in the reward.
- [`board_value_listener`](board_value_listener.md): win_prob_listener with a
  V that reads the board, fitted on simulated games.

The incumbent has been evaluated on the frozen suite once, against a Sonnet
spymaster (docs/log.md): 58% of games, sign p = 0.017.
