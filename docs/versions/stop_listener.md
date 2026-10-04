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
- **Search.** The frozen search with the assoc profile booster gives the
  200-clue shortlist. Each clue is re-valued at every number under the STOP
  listener: P(win) after the turn, exact over the own words found. STOP ends
  the turn like a neutral word, with nothing lost. No reply lookahead, and
  no k=1 tiebreak.

## Results

Judged by the user playing it (scripts/tools/play_server.py, "Stop
listener"). No suite run yet: the arena's gpt-oss guesser never stops, so
the suite would not see what this model changes.

The first version, on a trial model trained on 1,324 labels, still gives 4
on 26 of 30 boards. The number is only a cap for this listener, so a larger
one is nearly free (docs/log.md, "stop_listener").

## Open for the next model

- How the announced number is chosen, so that it says how many words the
  clue points at.
- `--set guess` (the strict set) against guess + stretch.
- A Sonnet check of the labels, when wanted.
