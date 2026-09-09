# z_threshold

A new baseline, not a replacement for any existing one -- registered
alongside `random`/`centroid`/`linear_scorer`/`oracle` in
`configs/spymasters.json` with `"roles": ["baseline"]`.

## What it is

For each candidate clue (`rarity_percentile <= max_rarity`, default 10.0,
in the `numberbatch` space by default), z-score its similarity to every
unrevealed board word against that clue's own mean/std across all 400
board words (`codenames/clue_stats.py`, new cached artifact
`cache/clue_stats.npz` + `cache/clue_stats_meta.json`, built by
`scripts/data/build_clue_stats.py`). Fixed percentile budgets
(`own_top=0.10`, `neutral_outside=0.10`, `opponent_outside=0.20`,
`assassin_outside=0.30`) convert once, at construction, into z thresholds
via `statistics.NormalDist().inv_cdf` (no scipy dependency).

A clue's intended own-word count `k` is however many own words clear the
own threshold, capped at `MAX_CLUE_NUMBER`. The clue is "valid" if every
remaining distractor stays under its own role's threshold too. Score is
`k` minus a risk term: for every non-own word, convert the z-margin
between it and the weakest intended own word back into cosine similarity
(multiply by the clue's own std -- the space `guesser_noise_std`'s
Gaussian noise actually lives in), get the probability that noise alone
lifts the distractor above the weakest intended word
(`torch.special.ndtr`), and weight that probability by
`abs(codenames.game.ROLE_REWARD[role])` -- so an assassin at a given
margin costs 10x a neutral at the same margin. See
`codenames/spymasters/z_threshold.py`'s module docstring for the full
derivation.

Two required fallbacks (never raises, never returns no clue):
1. No clue is fully valid -- score every eligible (`k >= 1`) clue
   ignoring the role thresholds.
2. No clue has any own word above the own threshold -- force the
   intended set to each candidate's single best own word (`k=1`).

Selection is fully deterministic: highest score, ties broken by margin
(`weakest - max(non-own z)`), then lowest clue-vocabulary index. No RNG
anywhere -- required by `docs/iteration-architecture.md` step 6's eval
store, which re-bills real LLM calls if a rerun's clue diverges.

## Why cache clue statistics separately

`codenames/arena.py` constructs a fresh spymaster inside every spawned
worker process (`docs/design-decisions.md`'s memory design note).
Recomputing each clue's mean/std over all 400 board words from the raw
similarity tensor would repeat an identical, model-independent
computation once per worker. `cache/clue_stats.npz` computes it once
(~1.6s over the full 111,440-clue vocabulary); `codenames/clue_stats.py`
loads it and validates its `clue_vocab_hash` against the live
`SimilarityTensor`, following the same pattern
`scripts/pipeline/featurize_rollouts.py` uses for rollout sets.

The full z-scored tensor is deliberately *not* also cached -- see
`codenames/clue_stats.py`'s module docstring for why that would
contradict the same memory discipline.

## Sanity check (10 real boards, `load_holdout_wordlist()`, seeds 0-9)

Every board's top pick announced `number=4` (the max). Checked this isn't
a degenerate "hub word" artifact by inspecting the raw (uncapped) own-word
count across the entire 111,440-clue vocabulary for a few boards: the true
maximum across the whole vocabulary was only 5-7 (never higher), and
hundreds to ~1,300 distinct clues cleared `k>=4` per board -- not one
runaway word dominating every board. With an 11,000-candidate pool (after
the rarity filter) and only a 90th-percentile own-word bar to clear, it is
apparently common for *some* clue to beat 4 of a random 9-word own set by
chance; `own_top` would need tightening (or `MAX_CLUE_NUMBER` raising) to
see non-4 picks more often on fresh boards. Chosen clues varied sensibly
per board and were often thematically plausible (e.g. `tank` for a board
whose own words included Missile/Battery/Soldier/Tube). Every board had
hundreds of finite-scoring (valid) candidates, so the fallbacks never
fired on these full, unrevealed boards -- they're exercised by
`tests/test_z_threshold_spymaster.py`'s synthetic fixtures instead, and
would be expected to fire more often on partially-revealed boards later
in a real game.

## Measured: safe for a listener that shares its space, dangerous otherwise

Real two-team self-play, 100 boards per row, synthetic guessers (no API
cost). This is the finding that matters most about this baseline:

| listener | assassin-hit | half-turns | mean clue number | correct/clue | own% |
|---|---|---|---|---|---|
| `noisy_numberbatch` (its own space) | **0.0%** | 5.27 | 3.14 | 2.91 | 94.2% |
| `noisy_glove` | **29.0%** | 6.73 | 3.10 | 1.62 | 69.6% |
| `noisy_wikipedia2vec` | **38.0%** | 6.75 | 3.18 | 1.49 | 65.9% |

Against a guesser using the same embedding space it selects on, this
baseline is excellent -- 0% assassin, 94% own-word rate, and it finishes
games in 5.3 half-turns. Against a guesser whose knowledge comes from a
*different* space it is worse than the centroid baseline was.

**The cause is structural, not a tuning problem.** The thresholds
guarantee the assassin is outside the top 30% *in numberbatch*, and
nothing constrains where it sits anywhere else. Measured over 40 boards,
for the clue actually chosen:

| space | assassin above z=0.52 | above z=1.28 |
|---|---|---|
| numberbatch (selected on) | 0.0% | 0.0% |
| glove | 25.0% | 7.5% |
| wikipedia2vec | 20.0% | 5.0% |

30% of boards put the assassin in the top 30% of a space the spymaster
never consulted, which lines up with the 29% assassin rate measured
against the glove listener. Concretely, seed 9 picks `counter` with the
assassin at z=-0.05 in numberbatch but **+1.79 in glove and +1.77 in
wikipedia2vec**.

This is `docs/design-decisions.md`'s "diversity must be in knowledge, not
noise" principle seen from the spymaster's side: a single-space spymaster
is only safe for a listener that happens to share its space. It matters
directly for the intended use as a fixed opponent, and for evaluation
against the LLM guesser, which is not a cosine ranker in any space and so
should be expected to behave more like the cross-space rows than the
first one.

The fix under consideration is to keep selecting on one space (the
positive signal stays simple) while requiring the *safety* thresholds --
assassin especially -- to hold in **every** built space. `ClueStats`
already carries mean/std for all three, so this is a filter change, not
an architectural one.

## Open for the next model

- `own_top`/`MAX_CLUE_NUMBER` interact in a way that makes "announce 4"
  the modal outcome on a fresh board under the defaults above -- not
  wrong, but worth a sweep before quoting this baseline's numbers as
  representative of typical play.
- The risk term assumes the guesser's only source of uncertainty is
  Gaussian noise in cosine-similarity space with a single fixed
  `guesser_noise_std` -- a deliberate simplification of
  `codenames/scorer.py::expected_reward_and_best_n`'s actual learned
  P(k, cause | clue), not a claim that real guessers behave this way.
- Only `space="numberbatch"` is exercised by the sanity check above; the
  other two spaces are equally supported by the constructor but untested
  against real boards here.
