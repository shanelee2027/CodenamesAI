# board_value_listener

win_prob_listener with a V that reads the board, not just the score.
Registry entry `board_value_listener` in `configs/spymasters.json`; code in
`codenames/spymasters/board_value_listener.py` and
`codenames/board_value.py`; correction `cache/board_value_sim.txt` with its
count table `cache/board_value_sim_counts.npz`
(`scripts/tools/eval_sim_value.py`, from games in `cache/sim_games.db` made
by `scripts/pipeline/simulate_games.py`).

## What changed

Only how each ending of the turn is valued. win_prob_listener values the
position after the turn by V(a, b), which sees how many words each side has
left and nothing else. Here the value is

    W = 1 − sigmoid( logit V(b', a') + board term(after-board) )

with the opponent to move. The exact cases stay exact: our board cleared is
1, revealing the opponent's last word is 0, the assassin is 0.

**The board term** is a LightGBM correction to the count logit, fitted on
turn states where the incumbent moves against win_prob_listener. These are
the positions this spymaster reads V at. Its features, per side, come from
the clue policy's clue × word tables:
- how clean the best 1-, 2-, 3- and 4-clue is;
- the hardest own word and the mean own word, each by its best one-word
  margin;
- how easily an own word and the assassin share a clue.

The games are simulated: the guesser is a fitted listener sampling rankings
(`codenames/guessers/listener_sample.py`, `sim:assoc`), so 12,000 games cost
compute, not money. The win_prob_listener side in them uses the incumbent's
booster with V1, the baseline this model is tested against.

**The after-board is approximated** by the likeliest one for each clue. The
j own words found are the clue's top j own words by listener score. A
neutral or opponent ending reveals the clue's top-scored word of that role.
The listener's distribution over endings is unchanged; only which words
those endings remove is fixed to the likeliest.

**Shared code touched.** `win_probability` (codenames/win_value.py) now
indexes W with `[..., :k]`, so W may be per clue. For the one-dimensional
tables every other spymaster passes, this is the same computation (its
tests pass unchanged). With a zero board term this spymaster gives
win_prob_listener's scores exactly (checked on three boards: maximum
difference 0.0).

**Parameters:** win_prob_listener's (`model_path`, `turn_model`,
`value_path`, `calibration_path`), plus `board_value_path` and
`board_counts_path`. On a board with a word outside the clue policy's tables
the term is 0, and this is win_prob_listener.

**Cost:** about 3,000 after-boards per move, scored in batches on the GPU:
3–4 s a move on a loaded machine.

## Results

**No gain in play** (docs/log.md, "Simulated games: a board-reading V").
Each test pairs this V with the incumbent's booster
(`model_path=cache/listener_gbt.txt`, `turn_model=frozen`), so the only
difference from win_prob_listener (V1, same booster) is the value function.
Both play learned_listener, and comparisons are paired by board:

| Test | boards | difference per game vs V1, 95% CI |
|---|---|---|
| simulated, the board model's training boards | 1,000 | +2.9 [+0.6, +5.1] (in-sample) |
| simulated, fresh boards | 600 | −0.6 [−3.8, +2.6] |
| gpt-oss suite | 87 | −1.2 [−8.1, +5.8] |

The correction predicts real outcomes better than the score alone (+0.0043
log-loss on 100k real states), but the gain does not carry into clue choice.
V1 stays the objective.

## Known risks

- **Selection bias.** The search takes the best of 200 clues, so it favours
  clues whose board-term errors flatter them. The turn calibration found the
  same bias in listener scores. The fit was checked on states that were
  actually played, not on after-boards chosen by a search.
- **The simulator under-kills on the assassin** (19 against gpt-oss's 31 for
  V1 on the suite). The correction may therefore undervalue assassin risk.
