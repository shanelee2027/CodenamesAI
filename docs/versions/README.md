# Model docs

One doc per model, named for the model — see `CLAUDE.md`'s naming rule
(descriptive names, not version numbers).

- [`learned_listener`](learned_listener.md): the incumbent.
- [`expected_words`](expected_words.md): the Gaussian baseline, and the
  incumbent's first stage.
- [`association_listener`](association_listener.md): the incumbent retrained
  with free-association counts, plus a pass priced on the absolute level they
  give.

The incumbent has been evaluated on the frozen suite once, against a Sonnet
spymaster (docs/log.md): 58% of games, sign p = 0.017.
