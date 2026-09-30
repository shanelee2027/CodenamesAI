# win_prob_listener

conceptnet_listener's listener with a new objective: the probability of
winning after the turn, not expected words minus role costs. Registry entry
`win_prob_listener` in `configs/spymasters.json`; code in
`codenames/spymasters/win_prob_listener.py` and `codenames/win_value.py`;
value table `cache/win_value.npz` (`scripts/data/build_win_value.py`).

## What changed

Only the objective. For each clue and number k, the listener gives the
distribution of how the turn ends:
- found k own words and stopped;
- or found j < k, then hit a neutral, opponent or assassin word.

Each ending leaves a score position, with a own words left, b opponent
words left, and the opponent to move. V(a, b) is the probability that the
side to move wins from there. The spymaster maximises

    P(win) = Σ over endings of P(ending) × W(ending)

- W(stop after k) = 1 − V(b, a − k), or 1 if the board is cleared.
- W(neutral after j) = 1 − V(b, a − j).
- W(opponent word after j) = 1 − V(b − 1, a − j), or 0 if it was the
  opponent's last word.
- W(assassin) = 0.

There are no role costs anywhere.

**V** comes from 10,266 recorded games (86,287 turn states) with gpt-oss
guessing, in which both sides played like the incumbent (the role-cost and
outside-option sweeps and the gpt-oss suite).
- Each cell is shrunk toward a logistic fit, then projected to be monotone
  (fewer own words left is never worse).
- V is blind to which words remain: this is design method A (docs/log.md,
  "win_prob_listener: design").
- Maximising against V is one step of policy iteration on the incumbent:
  the best turn now, assuming the incumbent's play afterwards.

**Parameters:**
- `model_path`: the booster, conceptnet_listener's by default.
- `turn_model`: `frozen` (Plackett–Luce with frozen scores) or
  `within_turn` (within_turn_listener's model of picks 2+).
- `value_path`: the V table.

The objective stays fixed while these vary, so listeners can be compared
without the reward confounding them.

**Other choices:**
- The k=1 tie-break tolerance (0.1 own words) is converted to win
  probability by what one own word is worth at the current score.
- The shortlist stage still ranks by its own cost-based reward.

## Results

**Clue choice** (600 validation positions, against conceptnet_listener: the
same listener, the old objective):
- mean number 2.38 → 2.74;
- clues for 4 rise from 105 to 187;
- nearly every change is upward.

**Games** (holdout_v1_gptoss against learned_listener, about $0.10 each):

| Challenger | win% | 95% CI | boards won both ways | sign p | assassin losses | mean k | own% |
|---|---|---|---|---|---|---|---|
| `turn_model=frozen` | **57.5%** | [0.50, 0.64] | **24 vs 10** | **0.024** | 25 vs 11 | 2.62 vs 2.33 | 77.5% vs 81.0% |
| `turn_model=within_turn` | 55.7% | [0.49, 0.63] | 22 vs 11 | 0.080 | 30 vs 9 | 2.43 vs 2.31 | 79.6% vs 81.8% |

The frozen arm played 93 boards and the within-turn arm 96. The rest were
discarded where gpt-oss would not rank.

**Reading.**
- **This is the first challenger to beat the incumbent significantly on the
  gpt-oss suite.** The frozen arm's p = 0.024 is 0.048 after a Bonferroni
  correction for the two arms.
  - The same listener under the old objective (conceptnet_listener) went
    24 vs 15 at p = 0.20.
  - So the objective, not the listener, was holding play back. This is what
    within_turn_listener's loss pointed to.
- **It wins by tempo and pays in assassins.** It hits the assassin more than
  twice as often (25 vs 11, Fisher p = 0.02) and still wins more. It is the
  same trade the role-cost sweep found at assassin cost 2.
  - The old reward priced the assassin at 10 words.
  - Here the assassin costs the current win probability, typically 0.3–0.6,
    the value of a few words.
  - Whether the listener also underestimates the assassin is open (below).
- **The within-turn model does not help under this objective.** It is 22 vs
  11 against the frozen arm's 24 vs 10. The two arms were not played against
  each other, and the difference is within noise.
  - It picks lower numbers (2.43) than the frozen arm (2.62), as its flatter
    later picks imply. It has the most assassin losses (30).

## Open

- **A second guesser.** A Nemotron run (about $0.86) would show whether the
  gain holds beyond gpt-oss. V was estimated from gpt-oss games, so the
  tempo it rewards is gpt-oss's.
- **Is the assassin rate a listener error or the right price?** Checked
  (`scripts/tools/check_turn_calibration.py`; docs/log.md).
  - It is mostly a price. The chosen clues were rated twice as risky as the
    incumbent's, which accounts for about 9.6 of the 14 extra assassin hits.
  - The rest is under-prediction: 25 hits against 18.9 predicted, p = 0.10.
    It is worst at 3–6% predicted risk (10 against 5.0, p = 0.03), the
    optimiser's curse.
  - Both listeners over-predict own words by about 0.13 per turn.
  - Next: recalibrate the turn-ending probabilities on recorded turns
    before the objective uses them.
- **Policy iteration step 2.** Re-estimate V from win_prob_listener's own
  games (it has played 186 so far), then play again.
- **Method B.** A V that reads the board: the assassin's neighbourhood, and
  how hard the leftover words are to clue.
- **A head-to-head of the two turn models** under this objective, to settle
  whether within-turn modelling helps once the reward values tempo.
