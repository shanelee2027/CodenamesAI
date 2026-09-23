# learned_listener

The incumbent. Registry entry `learned_listener` in `configs/spymasters.json`;
code in `codenames/spymasters/learned_listener.py`; derivation of the reward
in [`../clue-selection-learned.tex`](../clue-selection-learned.tex)
([pdf](../clue-selection-learned.pdf)).

## What it does

It replaces `expected_words`' assumed guesser (similarity plus Gaussian
noise) with a **learned** one, and keeps the search.

1. **Shortlist.** `expected_words` at σ=1.5 scores the whole ~11k-clue pool in
   one vectorised pass and keeps the top 200. Its job here is recall, not
   order.
2. **Listener.** For each shortlisted clue, a LightGBM model scores every
   unrevealed board word (`codenames/listener_features.py`, 44 features). A
   softmax over those scores is the probability the guesser picks each word
   next. The listener never sees roles; scores are split into ours and theirs
   only inside the reward.
3. **Reward.** Treating the scores as a Plackett–Luce model, the expected
   reward of announcing `k` has a closed form (`codenames/pl_reward.py`):
   each word gets an exponential clock, and the turn is decided by which own
   words ring before the first non-own one. Exact up to a 96-cell
   quadrature, and tested against brute-force enumeration.
4. **Pick** the legal (clue, k) with the highest expected reward.

## Parameters as deployed

| Parameter | Value | Meaning |
|---|---|---|
| `shortlist` | 200 | clues rescored by the listener |
| `sigma` | 1.5 | stage one only |
| `max_rarity` | 10.0 | clue-pool rarity filter |
| costs | neutral 0.2, opponent 1.0, assassin 10 | the spymaster's own prices (swept; nothing beat them) |
| `exclude_acronyms` | True | pool restriction |
| `outside_n` | 0 | outside option off (swept; lost monotonically to gpt-oss) |
| `k1_tiebreak` | off in the registry; on in the play server | not yet measured in games |
| booster | `cache/listener_gbt.txt` | `cache/listener_gbt_decoy.txt` is the decoy-trained variant |

## The listener

- **Teacher:** gpt-oss-120b at low effort, ~25k purchased rankings, many on
  deliberately bad clues (aimed at the opponent, the assassin, or nothing) so
  the listener learns what a guesser does with a clue that does not work.
- **Objective:** group softmax over each choice event, later picks
  down-weighted 0.75 per step, split by board seed, early-stopped on
  McFadden R².
- **Held-out quality:** McFadden R² 0.346 against 0.226 for the Gaussian it
  replaces, on boards nothing was ever selected on.
- **Evidence it uses:** three embedding spaces plus two more, human word
  associations in both directions (SWOW), language-model PMI, concreteness
  and WordNet. The association and PMI blocks carry most of the gain over
  embeddings alone.

## Results

Against Sonnet 5 (a transfer test, since the teacher is gpt-oss), fresh
seeds, both seatings:

| Opponent | Games | Win rate |
|---|---|---|
| `expected_words` σ=1.5 | 100 | 72% [0.63, 0.80] |
| `expected_words` σ=2.5 | 40 | 90% |
| `centroid` | 40 | 82% |

Not yet run on the frozen suite `holdout_v1`.

## Known limits

- The frozen-logit lookahead within a turn assumes independence of
  irrelevant alternatives, which the listener violates: removing one word
  flips its favourite about 20% of the time. The spymaster re-scores every
  turn, so this affects only the look-ahead inside one turn.
- The shortlist is chosen by the model being replaced; how often a deeper
  shortlist changes the pick has not been measured.
- The board-normalised softmax cannot tell a confident clue from a vague
  one. Decoy training pins that level (`listener_gbt_decoy.txt`), but using
  it through the outside option lost to gpt-oss, which never stops guessing.
