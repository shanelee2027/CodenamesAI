# pick_index_lookahead_listener

reply_lookahead_listener with the assoc booster and a pick-index model of our
turn. Registry entry `pick_index_lookahead_listener` in
`configs/spymasters.json`; code in
`codenames/spymasters/pick_index_lookahead_listener.py`; on the play server as
"Pick-index lookahead (win probability)".

## Why

It combines the parts that each measured best:

- **Listener features:** the assoc booster. It has the best win rate under
  the win objective, 59.1% against the incumbent (docs/log.md, "Progress
  notebook"). The assoc profile booster adds +0.0016 R², but there is no
  pick-index booster on its features.
- **How the guesser's later picks go:** the pick-index booster on the same
  56 features (`cache/listener_gbt_pick_indexassoc_depth9.txt`). Its Sonnet 5.5
  R² is 0.3374, against 0.3293 for the frozen assoc booster; the whole gain is
  at picks 2+. It has two extra inputs, the pick number j and j − k, so at pick
  j the guesser picks by softmax of S_j, its own scores for that pick and
  number. Nothing is done to the scores after the booster: it learned its own
  sharpness at each pick (val temperatures 1.04–1.07). The history model is
  more accurate (0.3573), but it needs about 130 booster runs per clue.
- **Decision rule:** P(win) after the turn with V(a, b), plus the one-move
  lookahead at the incumbent's reply (+3.9 points over the same rule without
  it, not significant).

## What it does

1. **Shortlist.** win_prob_listener's search ranks the shortlist with the
   frozen assoc booster, as before.
2. **Pick index on the top 8.** The best 8 clues are scored by the
   pick-index booster at every number k: k rows per word, one per pick. All
   of them go in one booster call. Each turn is enumerated exactly under those
   scores, stopping after k own words or at the first word that is not ours.
   Its P(win) is computed with V at each ending.
3. **Lookahead.** The best 6 (clue, number) pairs by that P(win) get
   reply_lookahead_listener's correction: the incumbent's reply on each
   probable after-board, centred on V by a per-score offset. The offset table
   is rebuilt under this turn model (`cache/reply_offset_pick_index.npz`,
   `scripts/data/build_reply_offset.py --spymaster
   pick_index_lookahead_listener`).

**Why only the top 8.** On 100 mid-game positions, a pick-index search over
the whole shortlist (about 200 clues at every number) picked the same clue
and number as the top-8 search 99 times. The one difference cost 0.0008
P(win). The full search's choice was already the frozen search's first clue
67% of the time, and within its top 8 99% of the time. The full search costs
about 2.7 s per move; the top 8 add about 0.2 s, for about 1.5 s per move in
all.

**Kept from reply_lookahead_listener:** the incumbent's reply. Its clue is
chosen by its own listener and objective, and its turn is played out under
our frozen assoc booster with V at the end. Our pick-index model is not used
for the opponent's turn.

## Results

All three rows are on the gpt-oss suite against the incumbent. Paired
differences are per game, with a 95% bootstrap over the boards both rows
played.

| Model | boards | win% | boards won both ways | sign p | assassin losses | mean k |
|---|---|---|---|---|---|---|
| win_prob, assoc booster | 99 | 59.1% | 28 vs 10 | 0.005 | 21 vs 10 | 2.59 |
| + reply lookahead | 91 | 56.0% | 25 vs 14 | 0.108 | 26 vs 8 | 2.53 |
| **+ pick index (this model)** | 92 | **57.6%** | **27 vs 13** | **0.038** | 27 vs 10 | 2.52 |

| Paired difference | per game, 95% CI |
|---|---|
| pick index over the frozen lookahead | +2.4 points [−4.7, +9.4] |
| the lookahead over plain win_prob (assoc) | −2.8 points [−10.0, +3.9] |
| this model over plain win_prob (assoc) | +0.0 points [−7.1, +7.1], 58.2% each on 91 boards |

**It beats the incumbent, but no more than plain win_prob on the same
booster does.**
- The pick index points up over the frozen lookahead.
- The lookahead points down on this booster, though it pointed up on the
  incumbent's (+3.9 points).
- All three differences are within noise. At about 90 boards a paired
  difference needs to be about 7 points to show.

The more accurate turn model changed the choice little in play. It changes
the clue in a third of positions (docs/log.md, "pick_index_lookahead_listener"),
but its mean number (2.52) is the same as the frozen lookahead's (2.53).
