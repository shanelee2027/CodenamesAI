# conceptnet_listener

isa_listener plus ConceptNet. Registry entry `conceptnet_listener` in
`configs/spymasters.json`; code in `codenames/spymasters/conceptnet_listener.py`;
booster `cache/listener_gbt_conceptnet.txt`; tables `cache/isa_sims.npz` and
`cache/conceptnet_sims.npz` (`scripts/data/build_isa_sims.py`,
`scripts/data/build_conceptnet_sims.py`).

## What changed

Seven features from ConceptNet 5.7's English edges are appended after
isa_listener's three WordNet is-a features. The blocks are `conceptnet` and
`compound` in `listener_training.FEATURE_BLOCKS`.

| Feature | Meaning |
|---|---|
| `cn_isa`, `cn_isa_rev` | ConceptNet IsA, word → clue and clue → word. 1 for a direct edge, 0.5 through one intermediate term. Kept apart from WordNet's `isa` because it is noisier ("moon IsA planet" is in it). |
| `cn_part` | PartOf, HasA or MadeOf, either direction (wheel/car). |
| `cn_typed` | The number of distinct specific relations joining them directly (UsedFor, AtLocation, ...; generic and lexical relations excluded). |
| `cn_any` | log(1 + max weight) of any direct edge. |
| `cmp_cw`, `cmp_wc` | "clue word" / "clueword" (resp. "word clue" / "wordclue") is a ConceptNet term: Donald Duck, firefly, William Shakespeare. |

NaN where the clue or word is not in ConceptNet. ConceptNet covers 99.8% of
clues, against 69.5% for WordNet nouns.

**Why, with numberbatch already in the tensor.** Numberbatch is built from
ConceptNet, but as an embedding, so it blurs every edge into a similarity. It
cannot say which relation holds, or whether the two words form a phrase.

The booster uses the same recipe and train split as isa_listener's.
Everything else is learned_listener's.

## Results

**Listener fit.** An ablation on identical rows
(`scripts/tools/eval_feature_blocks.py`), with each arm the 44 original
features plus the named blocks. The gains below are against the base arm,
with a paired bootstrap over boards (95% interval).

| Set | base R² | + is-a | + ConceptNet | + both (this model) |
|---|---|---|---|---|
| new boards | 0.3425 | +0.0023 [+0.0009, +0.0036] | +0.0007 [−0.0012, +0.0025] | **+0.0028** [+0.0008, +0.0047] |
| held-out words (gpt-oss) | 0.5414 | +0.0019 [+0.0002, +0.0036] | +0.0028 [+0.0010, +0.0048] | **+0.0040** [+0.0018, +0.0061] |
| held-out words (Sonnet) | 0.5456 | +0.0013 [−0.0000, +0.0028] | +0.0035 [+0.0016, +0.0055] | **+0.0052** [+0.0029, +0.0074] |

| Set | Accuracy, all picks (base → both) | Accuracy, pick 1 (base → both) | ECE top (base → both) |
|---|---|---|---|
| new boards | 0.4674 → 0.4723 | 0.6511 → 0.6532 | 0.0384 → 0.0403 |
| held-out words (gpt-oss) | 0.6408 → 0.6453 | 0.7915 → 0.7984 | 0.0802 → 0.0788 |
| held-out words (Sonnet) | 0.6293 → 0.6393 | 0.7562 → 0.7639 | 0.0538 → 0.0493 |

- The two sources add up: the gain from both is about the sum of the two
  alone.
- **Clues outside WordNet did not improve** (new boards 0.197 → 0.193). The
  phrase features were aimed at them. The gain is spread over ordinary clues
  instead.
- By split gain, `cn_any` and `cmp_cw` carry the ConceptNet block. Its
  is-a and part-whole features are nearly unused, and WordNet's `isa` does
  that job.

**Games** (holdout_v1_gptoss, 96 boards both ways, 192 games, about $0.10):

| Model | win% | 95% CI | boards won both ways | assassin losses | mean k | own% |
|---|---|---|---|---|---|---|
| conceptnet_listener | 54.7% | [0.48, 0.62] | 24 | 11 | 2.19 | 83.4% |
| learned_listener | 45.3% | [0.38, 0.52] | 15 | 10 | 2.27 | 82.1% |

Sign test p = 0.20. This is the largest lead over the incumbent of any
challenger so far (isa_listener 19 vs 15, pick_temperature 8 vs 19), but it
is not significant. Four boards are discarded because gpt-oss would not rank
a clue on them.

## Open

- A second guesser (Nemotron suite, ~$0.86) would say whether the lead
  holds beyond gpt-oss.
- The listener's pick-2 accuracy is 0.41 against a ceiling of 0.68–0.75
  (docs/log.md, "How much is left to explain"). Features of the single clue
  and word cannot close that gap; a model of picks within a turn can.
