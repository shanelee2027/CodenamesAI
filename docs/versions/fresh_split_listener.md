# fresh_split_listener

The thesis's GBT spymaster. It uses the expected reward of the Gaussian spymaster (`expected_words`), with the Gaussian model of the guesser replaced by a gradient-boosted listener. The listener is trained from scratch on a new three-way split of the online game's 800 words.

## What changed

**Data.** Nothing collected before 2026-11-15 is used. The 800 words were split uniformly at random into 400 training, 150 tuning and 250 test words (`board.py::load_three_way_split`, seed 20261116; the first draw was redrawn once because its tuning set held 39% base-game words against 47% overall). Positions were generated exactly as before (`collect_listener_data.py --vocab split_*`) and ranked by gpt-oss-120b at low effort: 29,659 training, 2,959 tuning and about 2,000 test positions.

**Roles of the three parts.** The boosters are fitted on the training words, with 15% of those boards for early stopping. Feature blocks are chosen by R² on the tuning words. The test words are used once: the final R² and the games.

**Features.** 72 columns: every listener feature except the blocks earlier models left out (`sense_*`, the two-hop `med_*`) and the clue's number `k`. On the tuning words (R², difference from the 73-column `full`, 95% bootstrap over boards):

| variant | R² | minus full |
|---|---|---|
| full | 0.3488 | |
| without the gpt-oss association features | 0.3343 | −0.0145 [−0.0175, −0.0114] |
| without GPT-2 PMI | 0.3435 | −0.0052 [−0.0073, −0.0032] |
| without the text embedding | 0.3445 | −0.0042 [−0.0068, −0.0017] |
| without all three | 0.3254 | −0.0233 [−0.0280, −0.0186] |
| without `k` (shipped) | 0.3489 | +0.0001 [−0.0008, +0.0011] |

The number does not change gpt-oss's pick order measurably, so the listener does not read it, and the spymaster's single prediction per clue is exact.

**Spymaster.** `learned_listener`'s search: the Gaussian model (sigma 1.5) shortlists 200 clues, the listener rescores them, and each clue's number maximises the expected reward under the game's costs. No cap on the number, as for the Gaussian spymaster in the thesis. No STOP model, overpromise charge, win value or tie-break.

## Files

- `cache/listener_gbt_fresh_split.txt` (+ `.meta.json`: online normalisation), a copy of `listener_gbt_fresh_split_no_k.txt`
- `scripts/pipeline/train_fresh_split_listener.py`
- `codenames/spymasters/fresh_split_listener.py`
- Gaussian sigma refitted on the training words: 2.00 (`cache/gaussian_sigma_split_train.json`); the reward baseline keeps sigma 1.5.

## Results on the test words

**[pending: test R², and games against the Gaussian spymaster at sigma 1.5 on test-word boards]**

## Open

- Positions use numbers 1-4 only; clues above 4 rely on the deepest guaranteed picks of 3-4.
- Everything that trains or tunes the listener is gpt-oss, as is the guesser in the games; a Sonnet copy of the test set would check how much of the model is gpt-oss specific.
