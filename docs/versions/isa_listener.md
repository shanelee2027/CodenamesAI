# isa_listener

The incumbent's spymaster with a listener that knows "is a kind of". Registry
entry `isa_listener` in `configs/spymasters.json`; code in
`codenames/spymasters/isa_listener.py`; booster `cache/listener_gbt_isa.txt`;
table `cache/isa_sims.npz` (`scripts/data/build_isa_sims.py`).

## Why

The category probe (docs/log.md, "Category clues: members against
associates") gave the incumbent three category members as its own words and
one associate of the category that is not a member: Orange, Lemon and Apple
with Pie for "fruit", Mercury, Saturn and Jupiter with Moon for "planet", and
so on over 10 cases.

- The incumbent's listener gave the associate 2–14% of first picks and ranked
  it above real members on average (Pie above at least one fruit).
- Averaged over the cases, the associate cut its probability that "clue 3"
  lands on the three members from 0.62 to 0.41.
- gpt-oss put the members first on 50 of 50 of the same boards.
- With the associate as the assassin, the clue's value went negative and it
  was never given.

No feature was directional. Wu-Palmer similarity scores the depth of a
shared ancestor, so Moon and "planet" (both celestial bodies) score well, and
every embedding is symmetric by construction.

## What changed

Three features, appended to `codenames/listener_features.py` as the
`taxonomy` block. All three come from WordNet's noun hierarchy, following
hypernyms and instance hypernyms:

| Feature | Meaning |
|---|---|
| `isa` | The board word is a kind of the clue. 1/(1+d), with d the fewest steps up from any sense of the word to any sense of the clue. 0 means there is no path; NaN means one side is not a WordNet noun. |
| `isa_rev` | The clue is a kind of the board word (poodle → Dog). |
| `isa_n` | How many candidates are a kind of the clue (a board constant). With three fruit present, a word that is not one is the associate, and the trees can explain it away. |

The booster is the current listener recipe (`listener_training.train`,
learning rate 0.01, early-stopped on val) with the new features added. It is
fitted on the same train split that `scripts/pipeline/train_listener_net.py`
uses, so the table below is exactly the deployed model.

Search, shortlist, reward and costs are learned_listener's.

**Existing boosters are untouched.** Features are only appended, and every
listener spymaster reads the columns its booster was fitted on, by name
(`listener_features.booster_columns`). The incumbent computes and scores
exactly as before.

## Results

**Listener fit.** Two boosters with the same recipe on identical rows, one
with every feature and one without the taxonomy block
(`scripts/tools/eval_isa_features.py`). Both are scored by
`train_listener_net.metrics`, the same function the neural listener was
scored with. The last column is restricted to events whose clue is a WordNet
category of at least one candidate (8–16% of events).

| Set | R² without | R² with | step-1 accuracy without | step-1 accuracy with | R² where isa_n ≥ 1, without → with |
|---|---|---|---|---|---|
| val | 0.3643 | 0.3645 | 0.679 | 0.678 | 0.484 → 0.487 |
| new boards | 0.3425 | 0.3448 | 0.651 | 0.659 | 0.447 → 0.460 |
| held-out words (gpt-oss) | 0.5475 | 0.5492 | 0.795 | 0.799 | 0.646 → 0.654 |
| held-out words (Sonnet) | 0.5456 | 0.5469 | 0.756 | 0.761 | 0.607 → 0.615 |

- The gain is small overall and larger where the features apply.
- Calibration is unchanged (ECE within ±0.004).
- Together the three features take 0.5% of the booster's split gain.

**Category probe** (`scripts/tools/category_probe.py`, 10 cases × 20 boards).
P(3 members) is the listener's probability that the first three picks are
exactly the three members. The same-recipe column separates the new features
from the newer recipe.

| | incumbent | same recipe, no is-a | isa_listener |
|---|---|---|---|
| P(3 members), associate present, mean of 10 cases | 0.41 | 0.43 | **0.57** |
| the same, on control boards | 0.62 | 0.63 | **0.74** |
| fruit / Pie | 0.25 | 0.36 | 0.53 |
| planet / Moon | 0.34 | 0.52 | 0.67 |
| animal / Africa | 0.14 | 0.11 | 0.35 |
| p(associate) at pick 1: Pie, Moon, Africa, Field | .14 .14 .08 .09 | .09 .09 .07 .17 | .07 .06 .04 .09 |
| gives the category clue with the associate neutral: planet, instrument, city | 1, 7, 6 of 20 | 7, 12, 4 | 10, 11, 9 |

- The recipe alone moves the average by 0.02 and the features by 0.14.
- gpt-oss put the members first on 50 of 50 of these boards. So even
  isa_listener still undervalues category clues, and it is still most wary
  when the associate is the assassin.

**Speed.** 0.72 s per clue against the incumbent's 0.55 s. The booster has
2,601 trees against 380.

**Games** (holdout_v1_gptoss, 94 boards both ways, 188 games, about $0.10):

| Model | win% | 95% CI | boards won both ways | assassin losses | mean k |
|---|---|---|---|---|---|
| isa_listener | 52.1% | [0.45, 0.59] | 19 | 13 | 2.18 |
| learned_listener | 47.9% | [0.41, 0.55] | 15 | 10 | 2.27 |

Sign test p = 0.61. Six boards are still discarded because gpt-oss would not
rank a clue on them.

**Reading.** isa_listener is level with the incumbent in games. The flaw it
fixes is real and measured, but boards where a clean category plus an
associate decides the game are too rare to move 94 boards. The fit gain is
+0.001–0.002 R² overall, and the category events are 8–16% of choices. A
Nemotron or Sonnet run was not made: at this effect size neither would
separate the models without far more boards.

## Open

- A human would give "animal 3" on the probe board, and isa_listener still
  rarely does. The listener spreads ~15% of first-pick mass over unrelated
  fillers even on control boards. That flatness is not category-specific, and
  it is the bigger remaining gap.
- ConceptNet IsA edges would cover categories WordNet lacks (slang, brands,
  "superhero"). WordNet covers 69.5% of the clue pool as nouns.
- It is untested whether the features help more under
  pick_temperature_listener's flatter later picks, where category clues for 3
  are exactly what is at stake.
