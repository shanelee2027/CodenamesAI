# within_turn_listener

conceptnet_listener with the guesser's picks 2, 3 and 4 of a turn modelled
as their own choices. Registry entry `within_turn_listener` in
`configs/spymasters.json`; code in `codenames/spymasters/within_turn_listener.py`
and `codenames/sequential_listener.py`; parameters `cache/sequential_listener.json`
(`scripts/pipeline/train_sequential_listener.py`).

## What changed

Only the reward. Every earlier listener spymaster scores a clue once and
treats pick 2 as pick 1 with the chosen word removed and the same scores
renormalised (Plackett–Luce with frozen scores). That has two consequences:
- pick 2 cannot depend on what pick 1 was;
- pick 2 is exactly as sharp as pick 1.

Both are measurably wrong:
- At pick 3 the favourite is predicted 52% of the time and picked 34%.
- The listener's pick-2 accuracy is 0.41 against a ceiling of 0.68–0.75
  (docs/log.md, "How much is left to explain").

At pick j ≥ 2 the logits become

    alpha_j * s(w) + beta_j * fit(w, picked)
    alpha_j = exp(a_j + b_j * (best score left − best score picked))

- `s` is conceptnet_listener's score, so pick 1 is unchanged.
- `fit` is the highest cosine between w and a word already picked, averaged
  over the embedding spaces that have both words.
- There are three parameters per pick (2, 3, and 4+), fitted by maximum
  likelihood on the val split's later picks: a = −0.24/−0.44/−0.58,
  b ≈ −0.07, beta = 5.2/5.9/4.1. A negative a means later picks are flatter.
  A positive beta means a word close to the one already picked is preferred
  (the "Cap then Glove" pull).
- A third term, same WordNet sense as a picked word, bought nothing once
  `fit` was in and is fixed at 0.

The reward stays exact. Every term depends only on the set of own words
picked, not on their order, so the subset DP of pick_temperature_listener
still applies with the logits recomputed per state
(`sequential_gain_and_penalty`; `tests/test_sequential_listener.py` checks it
against brute-force enumeration).

## Results

**Listener fit, later picks.** McFadden R², parameters fitted on val.

| Set | pick 2: frozen → this | pick 3: frozen → this | pick 4+: frozen → this |
|---|---|---|---|
| new boards | 0.268 → 0.303 | 0.147 → 0.194 | 0.078 → 0.133 |
| held-out words (gpt-oss) | 0.415 → 0.461 | 0.167 → 0.235 | 0.044 → 0.147 |
| held-out words (Sonnet) | 0.490 → 0.509 | 0.229 → 0.280 | 0.102 → 0.193 |

Calibration improves with it. On held-out words at pick 3, the favourite's
mean predicted probability falls from 0.52 to 0.42, against an actual hit
rate of 0.35.

**Clue choice** (`scripts/tools/compare_pick_temperature.py --models
conceptnet_listener within_turn_listener`, 600 validation positions):
- The same clue and number is chosen 56% of the time.
- Mean number falls from 2.38 to 2.10. Clues for 4 fall from 105 to 43, and
  every common change is a number going down (3→2 ×50, 4→3 ×38, 2→1 ×30).

**Games** (holdout_v1_gptoss, 95 boards both ways, about $0.10):

| Model | win% | 95% CI | boards won both ways | assassin losses | mean k | own% |
|---|---|---|---|---|---|---|
| within_turn_listener | 41.6% | [0.35, 0.49] | 11 | 5 | 2.00 | 84.7% |
| learned_listener | 58.4% | [0.51, 0.65] | 27 | 6 | 2.15 | 84.3% |

**Sign test p = 0.014. It loses, and loses clearly.** conceptnet_listener,
which is the same model with the old reward, led the incumbent 24 to 15 on
the same suite. So the reward change turned a lead into a significant loss.

**Why.** It is the same result as every lever that made the spymaster more
cautious:
- the role-cost sweep, where each higher price lost;
- the outside option;
- pick_temperature_listener, 8 vs 19.

gpt-oss guesses as many words as the clue number says, whatever its
confidence, so claiming fewer words just leaves them on the board. The
within-turn model is a better description of gpt-oss's ranking. But the
reward is myopic: it prices one turn's expected words and ignores tempo.
The old model's overconfidence in later picks was, in effect, paying for
that tempo. Correcting it removed the payment.

**The coherence term alone.** `fit` changes which clue wins, while the
temperature terms mostly lower the number. A "fit only" arm (beta alone,
with alpha fixed at 1) separates the two. It was saved with
`train_sequential_listener.py --arm "fit only"` to
`cache/sequential_listener_fit_only.json` and played with
`--challenger-param sequential_path=cache/sequential_listener_fit_only.json`.
It fits later picks about half as well: held-out pick-2 R² 0.428 against
0.461.

| Model | win% | boards won both ways | sign p | mean k |
|---|---|---|---|---|
| within_turn_listener, fit only | 49.0% | 17 vs 19 | 0.87 | 2.19 |
| learned_listener | 51.0% | | | 2.21 |

With k back at the incumbent's level, the loss is gone. So the flattening
terms, not the coherence term, cost the games. Coherence alone buys
nothing detectable either. It does not keep conceptnet_listener's 24–15
lead, but that difference is within the noise.

## Open

- **The reward, not the listener.** A within-turn model is the right
  description of the guesser, but it pays off only once the reward values
  tempo. That is the win-rate actor-critic direction: a learned value of
  the resulting position instead of a hand-priced turn.
- **Human guessers stop when unsure.** For them the flatter later picks may
  be exactly right, and the arena cannot show that.
