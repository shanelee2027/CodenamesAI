# stop_listener

win_prob_listener whose guesser can end its turn. Built after play-testing
pick_index_lookahead_listener, whose numbers counted near-random guesses that
the ranking-trained listener never stopped for ("obscure 4" for Alien,
Triangle, Spot, Sound). docs/log.md: "Relatedness labels" and
"stop_listener".

## What changed

- **Data.** Graded relatedness labels from gpt-oss (codenames/relatedness.py,
  prompt graded-v2): for each generated training position, the clue and the
  board without the number, sorted into "guess", "stretch" and the rest,
  unranked. The label set (guess + stretch by default) is ordered by
  gpt-oss's stored ranking of the same position.
- **Listener.** `cache/listener_gbt_stop.txt`
  (scripts/pipeline/train_stop_listener.py): at pick j the guesser picks
  among the remaining words and STOP. Word rows read the assoc profile
  booster's columns without `k`, plus the pick number. The STOP row reads
  only the clue-level columns and the pick number.
- **Number.** What the clue points at: the expected count of own words the
  guesser picks before STOP, with a wrong pick not ending the count,
  rounded, in [1, 4]. To this listener the number is only a cap, so choosing
  it by P(win) gave 4 almost always.
- **Search.** The frozen search with the assoc profile booster gives the
  200-clue shortlist. Each clue is valued at its own number under the STOP
  listener: P(win) after the turn, exact over the own words found. STOP ends
  the turn like a neutral word, with nothing lost. No reply lookahead, and
  no k=1 tiebreak.

## Results

Judged by the user playing it (scripts/tools/play_server.py, "Stop
listener"). No suite run yet: the arena's gpt-oss guesser never stops, so
the suite would not see what this model changes.

On a trial model trained on 1,324 labels, choosing the number by P(win) gave
4 on 26 of 30 boards. With the number set to what the clue points at, it
gives 3 on 15 and 4 on 15 (docs/log.md, "stop_listener").

On all 18,687 labels, over 30 boards (docs/log.md, "stop_listener"):

| STOP listener trained on | numbers | mean |
|---|---|---|
| guess + stretch (`cache/listener_gbt_stop.txt`, the default) | 3 on 7, 4 on 23 | 3.77 |
| guess only (`cache/listener_gbt_stop_guess.txt`, `stop_model_path`) | 2 on 15, 3 on 14, 4 on 1 | 2.53 |

Both are on the play server ("Stop listener" and "Stop listener, strict").

## Open for the next model

- Which label set becomes the default, once the user has played both.
- A Sonnet check of the labels, when wanted.
