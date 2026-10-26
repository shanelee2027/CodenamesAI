# margin_value_overpromise_listener

win_prob_overpromise_listener with one change: the win-value table V.

## Why

win_prob_overpromise_listener converts its cost per unattempted promised
word into win probability with `WinValue.one_word(a, b)`, the gain in P(win)
from one more own word at the current score. That is read off
`cache/win_value.npz`, which is shrunk toward a logistic fit and then made
monotone by isotonic regression. The projection pools neighbouring cells
into exact ties: 23 adjacent pairs in the 9 × 9 table. The one that matters
most is the opening. V(8, 9) = V(8, 8) = 0.637, so one_word(9, 8) = 0: on
the starting team's first clue the cost vanishes, and the plain P(win)
objective also sees no difference between finding 0 and 1 words. From team
B's side, after(8, 9) gave P(win) 0.522, 0.599, 0.599, 0.819 by own words
found, so a second word looked worth nothing.

At the opening the ties are not only the projection's doing: the two cells'
data (194 and 762 games) both give 63.7%, which they cannot distinguish,
while the logistic fit says the step is about 4 points.

## What changed

- **V** is `cache/win_value_margin.npz`:

      python scripts/data/build_win_value.py --max-game-id 15907 --margin 0.5 \
          --out cache/win_value_margin.npz

  - `--max-game-id 15907` uses the same 10,266 games as `cache/win_value.npz`
    (built 2026-09-30). The builder takes every finished game on the gpt-oss
    eval suite, and 4,034 more have been recorded since, by many different
    challengers, so a plain rebuild would change V's data as well as its
    shape. With `--margin 0` and this cutoff the rebuild matches
    `cache/win_value.npz` exactly.
  - `--margin 0.5` projects V − 0.5 × prior onto monotone tables and adds
    0.5 × prior back, so every step of V is at least half the logistic
    prior's step. The prior is strictly monotone, so nothing ties. Cells
    whose own steps are larger are unchanged; the game-weighted mean change
    from the shrunk data is 0.0014 (0.0005 for the plain projection).
  - Margin 1 was tried and rejected: it pushes corner cells outside [0, 1]
    (−0.011 and 1.025).
- **Effect on one_word:** one_word(9, 8) = 0.020 (was 0), one_word(8, 9) =
  0.064 (was 0.078). It is below 0.001 only where one word cannot change the
  game: 6–9 own words left against the opponent's last word (they win next
  turn either way), 9 against 2, and our last word against 8 or 9 (we win
  next turn either way). The opening value is small next to
  the ~0.15 typical later, which is what the smooth fit says: early, one word
  moves the win probability little.
- Everything else is win_prob_overpromise_listener's: the P(win) objective
  under the strict STOP listener, the cost `promise_cost` × (k − j) ×
  one_word, the number chosen by the objective. Config default
  `promise_cost` 0.5. `value_path` overrides the table.

## Results

**Opening clue** (no API; 30 holdout boards, first clue, mean number):
team A 3.53 / 3.23 / 3.10 at λ = 0 / 0.5 / 1 (old V: 3.53 at every λ), team
B 3.17 / 2.80 / 2.63 (old V: 3.10 / 2.57 / 2.47). The cost now acts on the
opening.

**Comfort arena** (vs assoc_profile_listener, two-step guess-only guesser,
100 boards both ways):

| model | win% | sign p | assassin (it vs opp) | comfort |
|---|---|---|---|---|
| margin V, λ = 0.5 | 51.0 | 0.86 | 15 vs 6 | 0.854 |
| old V, λ = 0.5 | 51.5 | 0.73 | 16 vs 4 | 0.851 |
| margin V, λ = 1.0 | 51.5 | 0.73 | 10 vs 7 | 0.883 |
| old V, λ = 1.0 | 53.5 | 0.28 | 11 vs 7 | 0.873 |

No measurable difference from the old-V model at either λ, and neither is
on the win%/comfort frontier. Removing the ties fixes the opening clue but
the opening is about 12% of turns and its one-word value is small, so the
games barely change. Kept as an exploration entry.
