# assoc_profile_listener

assoc_feature_listener with eleven more listener features from gpt-oss's
free associations. Registry entry `assoc_profile_listener` in
`configs/spymasters.json`; code in
`codenames/spymasters/assoc_profile_listener.py` and
`codenames/assoc_profile.py`; table `cache/assoc_profile.npz`
(`scripts/data/build_assoc_profile.py`); booster
`cache/listener_gbt_assoc_profile.txt`
(`scripts/pipeline/train_profile_listener.py`).

## What changed

Only the listener's inputs. Two blocks are appended to the 56 features of
the assoc booster:

- **The clue's profile** (8 features, the same for every board word). These
  can act only through interactions; for example, a vague clue should
  flatten the other features' effect.
  - How vague gpt-oss finds the clue, from its association lists: the share
    of distinct words, the mean pairwise overlap of the lists, and how often
    they open with the same word.
  - Its rarity percentile, concreteness, percent known, log frequency
    (Brysbaert norms) and number of WordNet senses.
- **Reverse associations** (3 per word). From each board word's own lists:
  the share that name the clue (plural and singular forms count), the mean
  reciprocal position of the clue in them, and that score's rank on the
  board.

**No new gpt-oss calls in play.** Lists exist for every clue in the pool
(rarity ≤ 10) and for all 400 board words, including the held-out ones.

**How it entered the pipeline.** The features are appended to
`FEATURE_NAMES`, and their table is registered in `APPENDED_TABLES`, as
every earlier block was. Older boosters read their own columns by name and
are unchanged (all tests pass; the incumbent does not load the table). The
table is not a clue × word similarity matrix, so it loads through its own
class (`TABLE_LOADERS`).

**One implementation.** `AssocProfile.columns` computes the features both
for training (the cached positions predate the columns, so they are
appended in the training script) and in play (`extract`). Checks:
- on all 39,250 cached positions it reproduces the exploratory columns of
  `scripts/tools/eval_extra_features.py` exactly (maximum difference 0, no
  NaN mismatches);
- a spymaster loading the booster computes the same columns on the gpt-oss
  suite's held-out boards (maximum difference 0).

The board-relative association block (A) tested alongside these was null
and is not included.

## Results

**Fit** (R², gain over the assoc booster, 95% bootstrap over boards):

| Set | assoc | assoc_profile | gain |
|---|---|---|---|
| val | 0.3707 | 0.3762 | +0.0054 [+0.0041, +0.0068] |
| new boards | 0.3499 | 0.3559 | +0.0059 [+0.0034, +0.0086] |
| held-out words, generated | 0.3340 | 0.3404 | +0.0064 [+0.0040, +0.0090] |
| held-out words (gpt-oss games) | 0.5637 | 0.5690 | +0.0053 [+0.0027, +0.0081] |
| held-out words, Sonnet | 0.5542 | 0.5621 | +0.0079 [+0.0054, +0.0106] |

Pick 1 gains +0.004 to +0.009. The new features take 7.6% of the trees'
gain.

**Play** (docs/log.md, "assoc_profile_listener"). Under the win-probability
objective (`win_prob_listener`, `model_path` set to this booster) against
learned_listener on holdout_v1_gptoss: 57.2%, boards won both ways 20 vs 7
(p = 0.019), on 90 boards. Paired with the assoc booster under the same
objective (28 vs 10), it is −1.1 points per game [−7.9, +5.6]: no detectable
difference. Not adopted; the assoc booster stays the listener for play.
