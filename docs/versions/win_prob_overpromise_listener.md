# win_prob_overpromise_listener

stop_listener's P(win) objective with overpromise_listener's cost for
promised words the guesser does not attempt. On the comfort arena's
scoreboard (docs/log.md, 2026-10-18) the frontier ran from win_prob (most
wins, least comfortable) to the strict overpromise models (most
comfortable), and the two ends are separate models: win_prob values a turn
by the chance of winning after it but never thinks about promises;
overpromise charges for broken promises but values the turn in net words.
This puts the charge on the win-probability objective.

## What changed

- **The value** is stop_listener's: the turn under the STOP listener (an
  exact recursion over the own words found, each pick a softmax over the
  words left and STOP), with P(win) after the turn from V(a, b) at every
  ending.
- **The cost.** A STOP after j picks on a clue for k is charged
  `promise_cost` × (k − j) × w, where w = `WinValue.one_word(a, b)` is what
  one more own word is worth now in win probability, at the turn's starting
  score (a own and b opponent words left). So `promise_cost` is in own words,
  as in overpromise_listener, and converted to win probability at the
  current score: a broken promise costs more where a word matters more. A
  wrong word is charged only through the state it leaves, and a turn that
  finds our last word is a win with no charge.
- **The number** is the objective's: the k in [1, 4] with the best value
  after the cost, as in overpromise_listener, not stop_listener's count of
  what the clue points at.
- **The STOP listener** defaults to the strict one
  (`cache/listener_gbt_stop_guess.txt`, trained on gpt-oss's "guess" group);
  `stop_model_path` overrides. The config default is `promise_cost` 0.5.
- On the play server: "Win probability + overpromise penalty 0.5, strict
  (guess only)" and "... penalty 1, strict". The value shown on /compare is
  P(win) minus the cost (`value_kind = "win_charged"`, labelled so).

**A property of V to know about.** V is smoothed and then made monotone by
isotonic regression, which leaves flat stretches. Where V(b, a − 1) =
V(b, a), `one_word` is 0 and the cost vanishes. The largest case is the
opening: the starting team's first clue (9 own words against 8) has
one_word = 0, so that clue is chosen exactly as with no cost. In the comfort
arena's games this is about 12% of a model's turns (the first turn of every
game it starts, and a few endgame scores such as 6 v 2 and 7 v 3).
Elsewhere one_word averages about 0.15.

## Results

**Numbers on 30 boards** (holdout vocabulary, seeds 0–29, the first clue;
no API). Team A opens with 9 own words against 8; team B is the same boards
from the other side, 8 against 9.

| model | team A: 1 | 2 | 3 | 4 | mean | team B: 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| stop, strict | 0 | 11 | 19 | 0 | 2.63 | 15 | 0 | 15 | 0 | 2.00 |
| overpromise 0.5, strict | 0 | 14 | 15 | 1 | 2.57 | 0 | 19 | 10 | 1 | 2.40 |
| overpromise 1.0, strict | 0 | 18 | 12 | 0 | 2.40 | 0 | 21 | 9 | 0 | 2.30 |
| this model, λ = 0 | 0 | 1 | 12 | 17 | 3.53 | 1 | 0 | 24 | 5 | 3.10 |
| this model, λ = 0.5 | 0 | 1 | 12 | 17 | 3.53 | 7 | 0 | 22 | 1 | 2.57 |
| this model, λ = 1 | 0 | 1 | 12 | 17 | 3.53 | 8 | 0 | 22 | 0 | 2.47 |

On team A's opening λ does nothing (one_word(9, 8) = 0, above). From team
B's side the number falls with λ, but never to 2: V is flat there too, and a
second own word is worth no win probability (P(win) by words found 0.522,
0.599, 0.599, 0.819). About 2 s per clue.

**Comfort arena** (docs/log.md, 2026-10-20; two-step gpt-oss guesser with
the strict "guess" set, against assoc_profile_listener, 100 boards × 2;
$0.16 for both runs):

| promise_cost | win% (95% CI) | boards both ways | sign p | assassin (it vs opp) | mean k (it vs opp) | comfort (95% CI) | stopped early (it vs opp), t, p |
|---|---|---|---|---|---|---|---|
| 0.5 | 51.5 [46.0, 57.0] | 18 vs 15 | 0.73 | 16 vs 4 (Fisher p 0.010) | 2.24 vs 2.25 | 0.851 [0.828, 0.872] | 25.3% vs 30.1%, −2.22, 0.029 |
| 1.0 | 53.5 [48.5, 59.0] | 19 vs 12 | 0.28 | 11 vs 7 (p 0.47) | 2.14 vs 2.24 | 0.873 [0.852, 0.892] | 21.9% vs 30.2%, −3.25, 0.0016 |

promise_cost 1.0 is on the win-vs-comfort frontier, between win_prob (54.5%,
0.756) and overpromise 0.5 strict (52.0%, 0.929), 0.7 points above the line
joining them. 0.5 is dominated by overpromise 0.5 strict and loses to the
assassin 16 to 4. Neither beats the incumbent on wins significantly; both
are more comfortable than it (fewer early stops), but much less so than the
net-words overpromise models.
