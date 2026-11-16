# fresh_split_listener

The thesis's GBT spymaster. It uses the expected reward of the Gaussian spymaster (`expected_words`), with the Gaussian model of the guesser replaced by a gradient-boosted listener. The listener is trained from scratch on a new three-way split of the online game's 800 words.

The model was settled in two steps on 2026-11-15, both under this name (the user's call: these choose the variant of the one standard GBT, they are not new models). The first listener was trained on rankings made with the clue's number shown; the shipped one is trained on rankings made without it, with a temperature per pick. "History" below keeps the first version's numbers.

## What it is

**Data.** Nothing collected before 2026-11-15 is used. The 800 words were split uniformly at random into 400 training, 150 tuning and 250 test words (`board.py::load_three_way_split`, seed 20261116; the first draw was redrawn once because its tuning set held 39% base-game words against 47% overall). Positions were generated as before (`collect_listener_data.py --vocab split_*`): a random board from the part's words, a random number of words revealed, a clue near one to three words of one identity or uniformly random. gpt-oss-120b at low effort was shown the clue **without a number** (`--no-number`) and ranked the unrevealed words, naming at least 80% of them. Each ranking gives its first n picks, n = our words still unrevealed (at most ceil(0.8 x words left), the depth the prompt guarantees); positions with none of our words left are dropped. 29,541 training positions (161,914 choice events), 2,948 tuning, 1,997 test.

**Roles of the three parts.** The boosters are fitted on the training words, with 15% of those boards (split by board) for early stopping. The feature blocks and the pick weights were chosen on the tuning words, and the temperatures are fitted on them. The test words are used once: the final R² and the games.

**Features.** 72 columns: every listener feature except the blocks earlier models left out (`sense_*`, the two-hop `med_*`) and the clue's number `k`. The blocks were chosen on the numbered rankings (History); on those, the number did not change gpt-oss's pick order measurably.

**Pick weights.** Pick j weighs 0.6^(j-1) in the fit. Chosen on the tuning words among 0.5, 0.6, 0.75, 0.9 and 1.0. Without temperatures, flatter weights win (later picks are overconfident under steep weights); with temperatures the five are within 0.001 R² on the no-number rankings and 0.003 on the numbered ones, and 0.6 is best or tied on both while keeping pick 1 sharpest:

| decay | tune R², no number, T = 1 | with temperatures | numbered rankings, with temperatures |
|---|---|---|---|
| 0.5 | 0.1626 | 0.1946 | 0.3548 |
| **0.6** | 0.1676 | **0.1953** | **0.3549** |
| 0.75 | 0.1741 | 0.1952 | 0.3539 |
| 0.9 | 0.1787 | 0.1945 | 0.3522 |
| 1.0 | 0.1816 | 0.1942 | 0.3519 |

**Temperature per pick.** With the trees frozen, pick j is softmax(s / tau_j) over the words left; tau fitted by maximum likelihood on the tuning words' no-number rankings (16,239 events), picks 7 and later sharing one: 1.03, 1.14, 1.37, 1.63, 1.86, 2.24, 2.90. Later picks are flatter, increasingly. Fitted first on the training words' early-stopping slice they came out sharper (0.94, 1.01, 1.16, 1.30, 1.40, 1.75, 2.39) and cost about 0.002 R² at pick 1 on the test words: the slice's boards are new but its words are training words, on which the trees are sharper. The decay table above uses those slice temperatures; it was chosen before the switch.

**Spymaster.** `learned_listener`'s search: the Gaussian model (sigma 1.5) shortlists 200 clues, the listener rescores them, and each clue's number maximises the expected reward under the game's costs. With a temperature per pick, pl_reward's exponential clocks no longer apply, so the expectation is computed exactly over which of our words have been picked so far (`pick_temperature_listener.tempered_gain_and_penalty`, 2^(own words) sets). No cap on the number, as for the Gaussian spymaster in the thesis. No STOP model, overpromise charge, win value or tie-break.

## Files

- `cache/listener_gbt_fresh_split.txt` (+ `.meta.json`: online normalisation), a copy of `listener_gbt_fresh_split_plain_no_k_d0.6.txt`
- temperatures: `PICK_TEMPERATURES` in the spymaster, from `cache/fresh_split_temps_plain_no_k_d0.6_fit_tune_on_plain_test.json` (scratchpad plain_temps.py --fit-on tune)
- `scripts/pipeline/train_fresh_split_listener.py --prompt plain --variant no_k --decay 0.6`
- `codenames/spymasters/fresh_split_listener.py`
- Gaussian sigma refitted on the training words: 2.00 (`cache/gaussian_sigma_split_train.json`); the reward baseline keeps sigma 1.5.

## Results on the test words

**R²**, the shipped listener with its temperatures (without them in brackets):
- no-number rankings (1,997 positions, 10,850 choice events): **0.2019** (0.1721); pick 1 0.5733, picks 2-4 0.1744, picks 5+ 0.0442 (−0.0228).
- numbered rankings (2,005 positions, 5,036 events), how games show clues: **0.3454** (0.3386); pick 1 0.5597, picks 2-4 0.1958. The numbered first version with its temperatures: 0.3426.
- with the slice temperatures instead: 0.1994 and 0.3426, pick 1 −0.0011 and −0.0019.

**Games** against the Gaussian spymaster at sigma 1.5 (no cap), 300 boards dealt from the test words, both seatings, gpt-oss guessing (label `gbt-tunetemps-vs-gauss15-test-p1`): **70.5% [67.0, 74.0]** (95% bootstrap over boards); boards won in both seatings 145 against 22. Assassin 3.7% against 8.2%; mean number 1.81 against 2.29; own words per clue 1.57 against 1.45; 89.2% of its team's guesses are its own words (Gaussian 74.7%). Numbers 1/2/3/4 = 34.2/52.6/11.1/2.1% of its clues, none above 4.

Same boards, other versions (paired over boards): with the slice temperatures 72.3% (this one −1.8 points [−4.8, +1.2]); the numbered first version 70.3% (+0.2 [−3.8, +4.3]). The flatter tuning-word temperatures make it more cautious still (mean number 1.98 -> 1.81, assassin 5.3% -> 3.7%), at no measurable cost in wins.

## History: the numbered listener (first version, 2026-11-15)

Trained on rankings made with the number shown (1-4), first `number` picks, pick weights 0.75^(j-1), no temperatures (`listener_gbt_fresh_split_no_k.txt`). 29,659 training, 2,959 tuning and 2,005 test positions.

Feature blocks on the tuning words (R², difference from the 73-column `full`, 95% bootstrap over boards):

| variant | R² | minus full |
|---|---|---|
| full | 0.3488 | |
| without the gpt-oss association features | 0.3343 | −0.0145 [−0.0175, −0.0114] |
| without GPT-2 PMI | 0.3435 | −0.0052 [−0.0073, −0.0032] |
| without the text embedding | 0.3445 | −0.0042 [−0.0068, −0.0017] |
| without all three | 0.3254 | −0.0233 [−0.0280, −0.0186] |
| without `k` (kept) | 0.3489 | +0.0001 [−0.0008, +0.0011] |

Test R² 0.3357 (pick 1 0.5594, picks 2+ 0.1796); with post-hoc temperatures 0.96, 1.05, 1.18, 1.43, 0.3426. Games against the Gaussian spymaster at sigma 1.5 on the same 300 test boards: 70.3% [66.9, 73.8]; assassin 4.8% against 8.5%; mean number 2.23 against 2.32; numbers of 5 or more in 4.3% of its clues, which then rested on the deepest guaranteed picks of 3-4. That gap is why the listener moved to the no-number rankings.

## Open

- Everything that trains or tunes the listener is gpt-oss, as is the guesser in the games; a Sonnet copy of the test set would check how much of the model is gpt-oss specific.
