# overpromise_listener

stop_net_words_listener with a cost for promising more than the guesser
attempts, and the number chosen by the objective again. The user's idea:
stopping is free for the game state, but a guesser who stops short is saying
the clue was not good.

## What changed

- **The cost.** When the guesser stops after j picks on a clue for k, the
  turn is charged `promise_cost` (λ) per promised word not attempted, so
  λ(k − j). A wrong word is charged only its role cost (the guesser attempted
  it). Announcing one more word costs λ times the chance the guesser stops
  before reaching it.
- **The number** is the k with the best expected net words after the cost,
  in [1, 4], instead of stop_listener's count of what the clue points at.
- Everything else is stop_net_words_listener's. `stop_model_path` picks the
  STOP listener (guess + stretch by default, `listener_gbt_stop_guess.txt`
  for the strict one). The config default is λ = 0.5.

## Results

The same 30 opening boards as the other stop listeners:

| STOP listener | λ | 2 | 3 | 4 | mean number |
|---|---|---|---|---|---|
| guess + stretch | 0 | 6 | 14 | 10 | 3.13 |
| guess + stretch | 0.25 | 8 | 13 | 9 | 3.03 |
| guess + stretch | 0.5 | 9 | 12 | 9 | 3.00 |
| guess + stretch | 1 | 9 | 13 | 8 | 2.97 |
| guess | 0 | 8 | 11 | 11 | 3.10 |
| guess | 0.25 | 13 | 15 | 2 | 2.63 |
| guess | 0.5 | 14 | 15 | 1 | 2.57 |
| guess | 1 | 22 | 8 | 0 | 2.27 |

For reference, on the same boards: the incumbent 2.93; stop_listener 3.77
(guess + stretch) and 2.53 (guess); stop_net_words_listener 3.53 and 2.50.

On the play server: "Overpromise penalty 0.5", "Overpromise penalty 0.5,
strict" and "Overpromise penalty 1, strict". About 2 s per clue. Judged by
the user playing.
