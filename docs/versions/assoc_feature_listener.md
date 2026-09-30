# assoc_feature_listener

conceptnet_listener plus gpt-oss's free associations as listener inputs.
Registry entry `assoc_feature_listener` in `configs/spymasters.json`; code
in `codenames/spymasters/assoc_feature_listener.py`; booster
`cache/listener_gbt_assoc_features.txt`; table `cache/assoc_sims.npz`
(`scripts/data/collect_associations.py --pool`,
`scripts/data/build_assoc_sims.py`).

## What changed

gpt-oss was shown each clue alone and asked for free associations, five
times at temperature 1. Two features come from those lists:

| Feature | Meaning |
|---|---|
| `assoc_share` | The share of the five lists that name the board word. Plural and singular forms count, and case, accents and punctuation are ignored. |
| `assoc_rank` | The mean over lists of 1 / the word's position (0 where a list does not name it). Named first counts far more than named 25th. |

- The lists cover the spymaster's whole clue pool (rarity ≤ 10), so the
  features exist for every clue it can give. That took 8,400 more lists
  (about $0.20).
- Only 0.83% of (clue, board word) pairs are named at all. The features are
  sparse and sharp.
- **Not association_listener.** That model used lists of this kind as a
  second training *target*. Here they are *inputs*.

The booster uses conceptnet_listener's recipe and train split plus the two
columns. Search, shortlist, reward and costs are learned_listener's.

A third feature block, WordNet sense agreement (`sense_agree`,
`sense_spread`), was tested alongside and left out because it bought nothing.

## Results

**Listener fit.** An ablation on identical rows
(`scripts/tools/eval_feature_blocks.py --arms senses_assoc`). R² gain over
conceptnet_listener's recipe, with a 95% paired bootstrap over boards:

| Set | conceptnet_listener R² | + assoc (this model) | + senses |
|---|---|---|---|
| new boards | 0.3453 | **+0.0046** [+0.0023, +0.0071] | −0.0006 [−0.0017, +0.0006] |
| held-out words (gpt-oss) | 0.5519 | **+0.0118** [+0.0088, +0.0148] | +0.0008 [−0.0003, +0.0019] |
| held-out words (Sonnet) | 0.5508 | **+0.0034** [+0.0007, +0.0062] | +0.0003 [−0.0007, +0.0014] |

| Set | Accuracy, pick 1 | ECE top |
|---|---|---|
| new boards | 0.6532 → 0.6560 | 0.0403 → 0.0358 |
| held-out words (gpt-oss) | 0.8050 → 0.8137 | 0.0767 → 0.0670 |
| held-out words (Sonnet) | 0.7639 → 0.7651 | 0.0492 → 0.0447 |

- This is the largest feature gain so far. On held-out words it is three
  times ConceptNet's.
- The gain on Sonnet's rankings is about a third of the gpt-oss gain. Part
  of the effect is gpt-oss predicting itself, but not all of it.

**Clue choice.** Against conceptnet_listener on 300 validation positions:
- the same clue and number 65% of the time;
- mean number 2.39 vs 2.41.

It picks different clues, not more cautious ones.

**Games** (holdout_v1_gptoss, 96 boards both ways, 192 games, about $0.10):

| Model | win% | 95% CI | boards won both ways | assassin losses | mean k | own% |
|---|---|---|---|---|---|---|
| assoc_feature_listener | 53.1% | [0.46, 0.60] | 23 | 10 | 2.19 | 84.6% |
| learned_listener | 46.9% | [0.40, 0.54] | 17 | 5 | 2.29 | 82.9% |

Sign test p = 0.43. This is about the same as conceptnet_listener's 54.7%
(24 vs 15) on the same suite. The better listener fit does not show up as a
larger lead at this sample size. The assassin count, 10 vs 5, is Fisher
p = 0.29.

**Second guesser** (holdout_v1_nemotron, the same 100 boards, about $0.86):

| Model | win% | 95% CI | boards won both ways | assassin losses | mean k | own% |
|---|---|---|---|---|---|---|
| assoc_feature_listener | 48.5% | [0.42, 0.55] | 13 | 15 | 2.22 | 79.8% |
| learned_listener | 51.5% | [0.45, 0.58] | 16 | 14 | 2.25 | 80.5% |

- Sign test p = 0.71: no lead at all under Nemotron.
- conceptnet_listener, which is this model without the association
  features, led 19 vs 10 on the same boards.
- So adding gpt-oss's associations lost the lead that conceptnet_listener
  held under a second guesser. The gpt-oss lead did not grow either.
- Both games results are noisy. Even so, the most direct reading matches the
  fit table: the associations mainly teach the listener gpt-oss in
  particular. The Sonnet R² gain was a third of the gpt-oss gain.
- **conceptnet_listener stays the better-supported challenger.**

## Open

- The listener keeps improving (is-a, then ConceptNet, then associations),
  but each game result sits in the same 52–55% band, never significant. Two
  explanations fit:
  - 100 boards cannot resolve a 3-point edge;
  - the myopic reward, not the listener, limits play (see
    within_turn_listener).

  A larger suite would separate them. So would playing the challengers
  against each other on the same boards.
- **Guesser-specific.** For a human guesser, gpt-oss's associations are a
  proxy at best.
