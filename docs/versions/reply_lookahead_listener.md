# reply_lookahead_listener

win_prob_listener that looks one move ahead: what does the incumbent do on
the board our clue leaves? Registry entry `reply_lookahead_listener` in
`configs/spymasters.json`; code in
`codenames/spymasters/reply_lookahead_listener.py`; offset table
`cache/reply_offset.npz` (`scripts/data/build_reply_offset.py`).

## Why

V(a, b) values the position after our turn by the incumbent's average win
probability at that score, the same for every board. But which words our
clue removes changes what the opponent can do next. Our unrevealed words are
its danger words: if our "Nile" is still up, its "river" clue is risky. If
our clue gets Nile found, we have unlocked its river cluster. A human
spymaster weighs this (Shane's framing), and V cannot see it.

A board-reading V (board_value_listener) tried to learn this from simulated
games and did not help in play. Here the opponent's next move is worked out
directly instead: the incumbent's behaviour is known exactly, since it is
code. Only the guesser is modelled, by our listener.

## What it does

1. **Shortlist.** win_prob_listener's search runs as before. Its best 8
   clues are scored at every number, and the best 6 (clue, number) pairs go
   on.
2. **Our turn.** Each pair's turn is enumerated under the frozen listener:
   exact Plackett–Luce picks, stopping at the first word that is not ours or
   after the number. Each outcome (own words found, ending word) is an
   after-board. The enumeration matches `pl_reward`'s turn model
   (tests/test_reply_lookahead.py).
3. **The opponent's reply.** On each after-board with P ≥ 0.01 (at most 12
   per pair), the incumbent's reply is worked out, and our value there is
   1 − its P(win) with that clue, minus a per-score offset (below). Other
   outcomes keep V's value.
4. **Pick.** The pair with the highest value is played.

**The reply, cheaply.** The incumbent's full search (0.56 s) runs once per
move, on the current board from its side. Its best 30 clues are kept with
its listener's scores. On an after-board the revealed words are dropped from
those scores, and the incumbent picks among the 30 by its own objective.
Checked against a full re-search on each of 1,924 after-boards from 40 real
positions (`scripts/tools/check_reply_lookahead.py`):

| Check | Result |
|---|---|
| Same incumbent clue | 66% (65% with the same number too) |
| Error in its P(win) | mean 0.016, 90th percentile 0.039 |
| **Our choice the same as with exact replies** | **95% of positions** |

Where the cheap reply misses the incumbent's clue, the two clues are close
in value.

**Centred on V.** On average the lookahead values after-boards lower than V
does, and more so the more own words our turn found (−0.006 with none, −0.054
with four). V is measured from real outcomes at each score; the lookahead is
a model with no such anchor. Used raw, it would penalise productive clues
for reasons unrelated to the board. So a per-score offset (mean lookahead −
V, from 103k outcomes on simulated training boards, shrunk toward 0) is
subtracted. V sets the level at each score; the lookahead contributes only
how this board differs from it. The offset is largest when we lead by one
word after our turn (−0.18 at a = 1, b = 2).

**Cost:** 0.72 s a move, all in. The win_prob_listener search alone takes
about 0.4–0.9 s.

**Parameters:** win_prob_listener's (frozen turn model, no calibration, no
outside option), plus `opponent_model_path` (the incumbent's booster),
`top_clues`, `candidates`, `opponent_clues`, `min_outcome`, `max_outcomes`
and `reply_offset_path`.

## Results

See docs/log.md, "reply lookahead".
