# association_listener

**What changed from `learned_listener`:** two things, separable by a parameter.

1. **The listener** is `cache/listener_gbt_assoc_w0.3.txt`: the incumbent's
   training recipe (same positions, features, split) plus a Poisson term on
   free-association counts at weight 0.3. gpt-oss listed 25 associations for
   each of 9,494 clues with no board shown, 5 times each. A board word's
   target is how many of the 5 lists name it. That target is not normalised
   over the board, so the scores gain an absolute level: exp(a·s + b) is the
   rate at which the word is named, with (a, b) in
   `listener_gbt_assoc_w0.3.assoc.json`.
2. **A pass priced on that level** (`pass_rate`, default 0.05). It is one
   outside option with a fixed score, that of a word named in `pass_rate` of
   lists, entering the reward as a zero-cost way for the turn to end
   (`codenames/pl_reward.py`). A vague clue loses much of its first pick to it;
   a sharp clue is barely touched. `pass_rate: null` turns it off.

**Why.** The board softmax cannot tell a clue that points strongly at one word
(4, 2, 2) from one that points weakly (2, 0, 0). See `docs/log.md`,
"Association counts".

**Measured so far** (validation boards, Sonnet set, gpt-oss holdout; no games):

| | board R² | Sonnet R² | gpt-oss holdout R² | level ρ, unseen clues |
|---|---|---|---|---|
| learned_listener | 0.355 | 0.553 | 0.342 | 0.41 |
| association_listener | 0.351 | 0.554 | 0.337 | 0.51 |

**First look at its clues.** On three boards, the retrained scores alone
(pass off) chose bolder clues than the incumbent ("marrying" for 4 where the
incumbent said "bible" for 3). With the 0.05 pass, the clues matched the
incumbent's.

**Open.**
- `pass_rate` is not swept.
- Against gpt-oss, which never passes, a pass is expected to cost games in
  the arena, as the decoy outside option did. The human study (`/eval`) and
  the qualitative comparison page (`/compare`) are where it can show.
- No frozen-suite games yet.
