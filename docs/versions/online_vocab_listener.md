# online_vocab_listener

floored_overpromise_listener with its listener and STOP listener retrained
for the online game's 800 words.

## Why

The online game (codenames.game) deals from 800 words. Every earlier booster
was trained on boards from the base game's 400 and had its z-scored features
normalised over those 400. The incumbent already transferred to the other 400
without a gap (R² 0.349 on online-only words against 0.340 on the old
held-out words, docs/log.md, 2026-10-30), so this is not a repair. It is more
training data on the words the model is now dealt, and z-scores measured over
the population it is dealt from.

## What changed

Only the two booster files. Search, reward, V, the overpromise charge and its
floor are floored_overpromise_listener's.

- **Data.** The 800 were split once (codenames/board.py::load_online_split):
  250 held-out words, drawn from the words that were not old training words,
  and 550 training words. gpt-oss ranked 9,980 generated positions on the 550
  (seeds from 8,000,000) and a test set of 1,886 on the 250. The listener's
  rows are the incumbent's own train and validation rows, by key, plus the new
  positions (15% of them, by board, go to validation).
- **Normalisation.** Every z-scored feature over the 800
  (`cache/clue_stats_online.npz`, `cache/extra_sims_online.npz`). The booster's
  sidecar `listener_gbt_online_vocab.meta.json` says so, and the bundle loads
  the online tables for it. A booster without a sidecar is on the base
  footing, so every older model loads as before.
- **Features.** The incumbent's 67. Two new blocks were built for this model
  and left out (below).
- **STOP listener.** `cache/listener_gbt_stop_online_vocab.txt`: the strict
  "guess" label set, the incumbent's labelled training positions plus the
  9,980 new ones (graded-v2 labels), on the new listener's columns and
  footing.

## The two blocks left out

Word kind (`proper_share`, `hub_assoc`, `hub_z`; codenames/word_kind.py) and a
text-embedding space (`te_z`, `te_rank`, `te_gaptop`; mxbai-embed-large).
Both were built to address what the word-difficulty test found: proper names
overrated, multi-word names without features, many-meaning words such as
CODE underrated. Each variant was fitted on identical rows. Gain in R² over
the 67-column booster, 95% paired bootstrap over boards:

| Set | + kind | + text | + both |
|---|---|---|---|
| online held-out words, gpt-oss | +0.0049 [+0.0025, +0.0073] | +0.0045 [+0.0018, +0.0072] | +0.0081 [+0.0046, +0.0116] |
| held-out words, generated, Sonnet 5.5 | −0.0042 [−0.0064, −0.0020] | −0.0023 [−0.0048, +0.0002] | −0.0035 [−0.0066, −0.0005] |

On the clean Sonnet 5.5 set (online held-out words, bought afterwards) the
picture is different: kind −0.0012 [−0.0035, +0.0010], text +0.0011
[−0.0015, +0.0037], both +0.0005 [−0.0030, +0.0040]. Neither block helps or
hurts Sonnet there. The harm on the old set was contamination: the kind
columns are per-word constants, and half of that set's words are training
words, on which the booster can learn gpt-oss's taste for the word itself.

Thus the blocks fit gpt-oss better and leave Sonnet unchanged: what they add
is gpt-oss's own taste, not how a guesser reads a clue. This is plausible for
`hub_assoc` in particular, which is computed from gpt-oss's own association
lists. Listeners are judged on Sonnet 5.5, which cannot tell the four apart,
so the simplest is shipped. The text block has the best point estimate and is
the one to revisit. Their boosters are kept in `cache/`
(`listener_gbt_online_vocab_kind.txt`, `_text.txt`, `_kind_text.txt`).

## Results

**Listener fit** (`scripts/pipeline/train_online_vocab_listener.py`). R²
against the incumbent's booster on the same positions, each on its own
footing, with the gain and a 95% bootstrap over boards:

| Set | Incumbent | This model | Gain |
|---|---|---|---|
| online held-out words, generated (gpt-oss; clean for both) | 0.3315 | 0.3368 | +0.0053 [+0.0028, +0.0077] |
| online held-out words, generated, Sonnet 5.5 (clean for both) | 0.3472 | 0.3499 | +0.0027 [+0.0002, +0.0050] |
| held-out words, generated, Sonnet 5.5 | 0.3308 | 0.3364 | +0.0057 [+0.0028, +0.0087] |
| held-out words, generated (gpt-oss) | 0.3408 | 0.3522 | +0.0115 [+0.0088, +0.0143] |
| online words, generated | 0.3489 | 0.3598 | +0.0109 [+0.0081, +0.0137] |
| new boards (training words) | 0.3559 | 0.3574 | +0.0016 [−0.0005, +0.0037] |

The first two rows are the clean tests: 250 words neither booster was trained on.
The other held-out sets are partly contaminated for this model. 74 of the 150
old held-out words are online training words, and so are many of the 424
online-only words. The old Sonnet 5.5 set has the same contamination, and it
inflates the gain there (+0.0057 against +0.0027 on the clean Sonnet set).

**STOP listener.** On the 1,520 online validation positions, which neither
model was trained on: R² 0.4768 against the incumbent's 0.4689 (by pick
0.589/0.430/0.324/0.498 against 0.581/0.418/0.316/0.496). It is calibrated
(predicted stop rate 0.357, actual 0.354).

**Play.** It gives the same clue as floored_overpromise_listener on 13 of 20
boards from the online 800 (seeds 11-30). Where the two differ, it is mostly
another clue for the same number (TRANSPORT for DRIVING, WHITE for WEDDING).
No arena run: a listener gain of this size is far inside the noise of a
200-game arena.

## On the play server (2026-11-01)

At the user's request this model replaced floored_overpromise_1 as the play
server's default (`online_vocab_1`; the public page serves only the default).
The other entries built on the old listener got online-words twins
(`<key>_online`, `scripts/tools/play_server.py::ONLINE_TWINS`): the same class
and parameters with the listener and STOP listener swapped for the retrained
ones. The originals stay, so their results still reproduce, and the frozen
arena keeps assoc_profile_listener as its opponent. Not twinned:
profile_temperature (its per-pick temperatures were fitted to the old
booster), win_prob_assoc and pick_index_lookahead (another booster).

The twins of the guess + stretch entries (stop, stop_net, overpromise) needed
that STOP listener retrained too: `cache/listener_gbt_stop_stretch_online_vocab.txt`,
same recipe (`train_stop_listener.py --set guess+stretch --base-booster
listener_gbt_online_vocab.txt --normalization online --with-online`), val R²
0.3039 against the old one's 0.2959 (different validation sets), predicted
stop rate 0.190 against an actual 0.199.

## Open

- An arena test against the frozen opponent, if a decision-rule change is
  built on it.
