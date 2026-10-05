# stop_net_words_listener

stop_listener with the incumbent's reward: each clue is valued by the
expected net words of its turn, not by P(win) after it. Asked for by the
user, to compare the two objectives under a guesser that can stop.

## What changed

Only the value of a turn (codenames/spymasters/stop_net_words_listener.py):
+1 per own word, minus the role cost of the word that ends the turn
(0.2 neutral, 1 opponent, 10 assassin by default; `neutral_cost` etc.
change them), and 0 for STOP. It is exact over the own words found, like
stop_listener's P(win).

Kept from stop_listener:
- the 200-clue shortlist, from its frozen P(win) search (wide, and every
  clue on it is rescored, so the objective there matters little);
- the STOP listener (`stop_model_path`: guess + stretch by default,
  `cache/listener_gbt_stop_guess.txt` for the strict one);
- the number, set to what the clue points at;
- no k=1 tiebreak, though it is in this model's units.

## Results

On 30 boards, through the play server:

| STOP listener | numbers | mean | P(win) version's mean |
|---|---|---|---|
| guess + stretch ("Stop listener (net words)") | 2 on 3, 3 on 8, 4 on 19 | 3.53 | 3.77 |
| guess ("Stop listener, strict (net words)") | 2 on 16, 3 on 13, 4 on 1 | 2.50 | 2.53 |

Mostly the same clues as the P(win) versions (cooking, breathing, branch,
procedure, castle on the first boards). Judged by the user playing.
