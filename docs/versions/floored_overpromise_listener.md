# floored_overpromise_listener

margin_value_overpromise_listener with a floor under the overpromise cost.

## Why

Playing the margin-V model at penalty 1 on the play server, the user found
it too cautious early and too aggressive when behind, with clues that made
little sense. The cost per promised word the guesser does not attempt is
`promise_cost` × one_word(a, b), what one more own word is worth now in win
probability. one_word is near 0 where one word cannot change the game, mostly
two or more words behind with the opponent close to winning (one_word(5, 2)
= 0.006, one_word(7, 3) = 0.007 on the margin V). There the cost vanishes,
and the P(win) objective, which already favours long shots when behind, can
promise numbers its own guesser will not attempt at no charge.

A constant cost was the other option. It would have to sit near the
middle-game value of one_word (~0.15) to matter, seven times the opening's
0.020, so it would make the opening clue smaller still, the opposite of what
was wanted there.

## What changed

- **The charge** is `promise_cost` × max(one_word(a, b), `charge_floor`).
  Every position where one_word is above the floor is unchanged.
- **Default** `charge_floor` 0.05 and `promise_cost` 1.0, the penalty the
  user was playing.
- Everything else is margin_value_overpromise_listener's: V from
  `cache/win_value_margin.npz`, the strict STOP listener, the P(win)
  objective, the number chosen by the objective.

## Results

**Clue number by score** (no API; λ = 1; 12 holdout boards per score, team
A to move, the given numbers of own and opponent words revealed at random
plus 0–3 neutrals; two boards appear in both (4, 1) and (4, 2)). Mean
number; a = own words left, b = opponent's, gap = a − b (> 0 behind).

| (a, b) | gap | incumbent | floor 0 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|
| (9, 8) | 1 | 3.25 | 3.08 | 3.08 | 3.00 | 2.58 |
| (4, 1) | 3 | 2.25 | 4.00 | 1.83 | 1.00 | 1.00 |
| (4, 2) | 2 | 2.25 | 2.83 | 2.83 | 2.75 | 2.67 |
| (4, 3) | 1 | 2.17 | 2.25 | 2.25 | 2.25 | 2.25 |
| (4, 4) | 0 | 2.25 | 1.92 | 1.92 | 1.92 | 1.92 |
| (4, 5) | −1 | 2.25 | 1.83 | 1.83 | 1.83 | 1.83 |
| (6, 3) | 3 | 3.17 | 3.33 | 3.33 | 3.08 | 2.75 |
| (6, 4) | 2 | 3.25 | 3.08 | 3.08 | 3.00 | 2.83 |
| (6, 5) | 1 | 2.67 | 2.17 | 2.17 | 2.17 | 2.17 |
| (6, 6) | 0 | 2.50 | 2.00 | 2.00 | 2.00 | 2.00 |
| (6, 7) | −1 | 2.58 | 1.92 | 1.92 | 1.92 | 1.92 |

Floor 0 is margin_value_overpromise_listener at λ = 1.

- **Opponent on its last word, (4, 1).** Floor 0 promised 4 on 10 of 12
  boards, mostly with clues that do not connect four words: BLANK,
  GOVERNMENT, FASCIST, ABUSE, INCLUSIVE, FICTIONAL. These are the all-in
  clues the floor is for. At 0.02 three remain (BLANK, WATCHES, INCLUSIVE);
  at 0.05 none, and the clue becomes a safe 1.
- **Two words behind, (4, 2).** The model still says 3 where the incumbent
  says 2, with weak clues (TONE, DARLING, GENUS); one_word is 0.031 there, so
  the floor does not reach it. That behaviour comes from the P(win)
  objective itself, not from the cost vanishing.
- **Level or ahead,** the model is about 0.3–0.6 below the incumbent at
  every floor. Its caution is not only on the opening; it comes from the
  strict STOP listener and the objective, which the floor does not touch.
- **The opening** goes from 3.08 to 3.00 at 0.05 (one board of twelve); 0.1
  would give 2.58.

**Comfort arena** (2026-10-28; two-step gpt-oss guesser with the strict
"guess" set, assoc_profile_listener as the fixed opponent, 100 boards × 2
games, none discarded; `floored_overpromise_listener:646834c252ac`, 776 s,
about $0.08). Win% and comfort with board-bootstrap intervals from
`plot_comfort_frontier.py`.

| model | win% (95% CI) | boards both ways | sign p | assassin (it vs opp) | mean k (it vs opp) | comfort (95% CI) |
|---|---|---|---|---|---|---|
| margin V + overpromise 1.0 (floor 0) | 51.5 [46.0, 57.0] | 18 vs 15 | 0.73 | 10 vs 7 | 2.12 vs 2.21 | 0.883 [0.866, 0.901] |
| **floored, 0.05** | 51.5 [46.0, 57.5] | 18 vs 15 | 0.73 | 9 vs 8 | 1.97 vs 2.20 | 0.912 [0.898, 0.924] |

Same wins, more comfort (+0.029; the two intervals barely touch), and a
lower mean number (1.97), which is the long-shot numbers the floor removes.
Still not on the frontier: overpromise 0.5 strict has 52.0% at 0.929.
