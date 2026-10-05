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
- [`assoc_profile_listener`](assoc_profile_listener.md): assoc_feature_listener
  plus the clue's association profile and reverse associations as listener
  features.
- [`reply_lookahead_listener`](reply_lookahead_listener.md): win_prob_listener
  that works out the incumbent's reply on the board each clue leaves.
- [`pick_index_lookahead_listener`](pick_index_lookahead_listener.md):
  reply_lookahead_listener with the assoc booster and a pick-index model of
  our turn.
- [`stop_listener`](stop_listener.md): win_prob_listener with a listener
  that can end its turn, trained on which words gpt-oss says a clue points at.
- [`stop_net_words_listener`](stop_net_words_listener.md): stop_listener
  valued in expected net words (+1, -0.2, -1, -10) instead of P(win).

The incumbent has been evaluated on the frozen suite once, against a Sonnet
spymaster (docs/log.md): 58% of games, sign p = 0.017.
