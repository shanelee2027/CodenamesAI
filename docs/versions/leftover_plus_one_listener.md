# leftover_plus_one_listener

online_vocab_listener that knows the guesser's +1 can go back to the previous
clue.

## Why

Codenames allows one guess beyond the number, and human guessers spend it on
a word an earlier clue was meant for and the team missed. Every model so far
values a turn as if the +1 did not exist. So a leftover from the last clue is
worth nothing to it until it is clued again, and it re-clues leftovers it
could have left to the +1. The history probes (docs/log.md, 2026-11-01) found
that gpt-oss can be prompted into this behaviour only at medium effort, and
that the behaviour is hard to learn as a listener term. So we write the rule
down instead, and put it into the objective. Nothing is retrained.

## The rule

The user's, written for the strict-subset guesser of the comfort arena:
1. Only our previous clue counts.
2. It left something only if its turn was cut short by a wrong guess.
3. Unfound: its number, minus the own words that turn found, minus the own
   words of its group that the opponent has revealed since (the guesser sees
   "your word" turn up and counts it as found). Nothing to do at 0.
4. If this turn ends with nothing wrong and own words still to find, one more
   guess goes back to the previous clue.

## The replay (free)

First we applied the rule to the comfort arena's 3,398 recorded games
(`scripts/tools/replay_leftover_plus_one.py`). The group is the guesser's
stored "guess" set and ranking for the previous clue, read from the store
with the API disabled. Each turn was replayed alone: the game as recorded,
plus the one guess the rule would add.

| | turns | of 25,258 |
|---|---|---|
| previous turn cut short by a wrong guess | 5,327 | 21.1% |
| the group had words after the wrong guess | 3,524 | 14.0% |
| a leftover to go back to | 3,428 | 13.6% |
| this turn clean, so the +1 is taken | 1,914 | 7.6% |

The +1's word was ours 49.0% of the time, a neutral 32.5%, the opponent's
14.9% and the assassin 3.6%. A leftover is a word the guesser ranked below a
word that turned out wrong, so it is weak. Taken every time, the +1 is about
even: per game +0.28 own words, +0.08 opponent words and +2.0% of games lost
to the assassin.

The vagueness check fixes this. By the STOP listener's probability for the
leftover at the previous clue's next pick (over the board now, with STOP):

| p(word) | +1s | ours | neutral | opponent | assassin |
|---|---|---|---|---|---|
| 0.0–0.1 | 851 | 0.221 | 0.432 | 0.271 | 0.075 |
| 0.1–0.2 | 312 | 0.333 | 0.503 | 0.151 | 0.013 |
| 0.2–0.3 | 162 | 0.506 | 0.444 | 0.043 | 0.006 |
| 0.3–0.4 | 149 | 0.886 | 0.114 | 0 | 0 |
| 0.4–0.5 | 159 | 0.962 | 0.038 | 0 | 0 |
| 0.5–1.0 | 281 | 0.993 | 0.007 | 0 | 0 |

Thus a leftover the listener still rates highly is almost always ours, and
one it rates low is mostly not. The model uses this probability directly.

## What changed

Only the value of a turn's clean endings. Search, number choice, listener,
STOP listener, V and the floored overpromise charge are online_vocab_listener's.

Where the turn ends cleanly (at STOP, or at the number with every guess ours)
and the rule finds a leftover, the ending is worth

    q * [sum_w p(w) * value after w  +  p(STOP) * value after the turn]
      + (1 - q) * value after the turn

instead of the value after the turn. p is the STOP listener's next pick for
the previous clue (pick j + 1 after the j guesses it got) over the words on
the board and STOP, renormalised without the own words this turn found. An
own word found by the +1 ends the turn one word better off. A bad one ends it
as a wrong guess does. STOP is the guesser passing. The overpromise charge at
a STOP ending is unchanged: the +1 attempts no word of this clue.

q = `plus_one_rate` (1) times the chance that unfound is still above 0. The
spymaster cannot know which of our words the opponent revealed were the old
clue's; each counts as the clue's with probability p0(w), the STOP listener's
next pick for the previous clue over the board as our previous turn left it.

History comes in through a new defaulted field, `TurnContext.history`
(`PastTurn`: whose turn, clue, number, guesses in order). This is the one edit
outside the model's own files. Callers that do not fill it (the arena's game
loop) get online_vocab_listener exactly.

## Results (free)

`scripts/tools/compare_leftover_clues.py`, 60 arena positions with a leftover
by the rule: the clue differs from online_vocab_listener's on 31. The clue's
targets include the leftover the listener rates highest on 19 positions,
against 34 for online_vocab_listener. So the model re-clues leftovers less
often and leaves them to the +1.

The cost is visible in the clues. To keep the turn clean for the +1 it
sometimes plays a safe 1 (HAY 1, SOUR 1, PUPPY 1) where online_vocab_listener
re-clued the leftover with a 2 or 3. Whether that pays depends on whether the
guesser really takes the +1. The arena's guesser never does, so no arena run
can show the gain. It needs a guesser that follows the rule (the replay's
rule, played forward), or humans.

## On the play server (2026-11-01)

Entry `leftover_plus_one_1`, now the default and so the public page's model.
Both pages send the game's turns with every clue and explain request, and the
explanation's P(win) is computed with them. A clue takes about 2 s with a
history, against about 1 s without.

## Open

- A rule-following arena guesser (the strict-subset guesser plus the
  leftover +1, gated or not) and a run against the frozen opponent.
- q for humans: unknown. Logging human games on the public site would
  measure it (a privacy decision for the user).
