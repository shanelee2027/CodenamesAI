"""The GBT spymaster of the thesis: learned_listener's expected-reward search
with a listener trained from scratch on a three-way split of the online
game's 800 words (docs/versions/fresh_split_listener.md).

**The listener.** `cache/listener_gbt_fresh_split.txt`
(scripts/pipeline/train_fresh_split_listener.py --prompt plain --variant no_k
--decay 0.6, a copy of `listener_gbt_fresh_split_plain_no_k_d0.6.txt`): 72
features, fitted only on gpt-oss rankings of positions generated on the
split's 400 training words, with every z-score over the 800 (sidecar
`listener_gbt_fresh_split.meta.json`). gpt-oss was shown the clue without a
number and ranked the board; each ranking gives its first n picks, n = our
words still unrevealed, pick j weighted 0.6^(j-1) in the fit. The feature
blocks and the decay were chosen by R² on the 150 tuning words; the 250 test
words are for the final evaluation only. It does not read the clue's number,
so one prediction per clue serves every number.

**Temperature per pick.** At pick j the guesser is modelled as choosing from
the words left with probabilities softmax(s / tau_j), PICK_TEMPERATURES
below, picks 7 and later sharing the last. They were fitted by maximum
likelihood with the trees frozen, on the tuning words' no-number rankings,
and lift test R² from 0.1721 to 0.2019 on the no-number rankings and 0.3386
to 0.3454 on the numbered ones. Fitted instead on the training words'
early-stopping boards they came out sharper (0.94 at pick 1) and cost a
little at pick 1 on new words: the trees are sharper on words they have
seen, even on new boards. Because the scores' scale changes with the pick,
pl_reward's exponential-clock formula no longer applies; the expected reward
is computed exactly over which of our words have been picked so far
(pick_temperature_listener.tempered_gain_and_penalty).

**The search** is learned_listener's: the Gaussian expected-reward model
shortlists 1000 clues, the listener rescores them, and each clue's number is
the one with the highest expected reward under the game's costs. No
overpromise charge, STOP model, win value or tie-break: the expected reward
alone, so the one difference from the Gaussian spymaster is the model of the
guesser. **No cap on the number** (`max_number=None`), as for the Gaussian
spymaster in the thesis.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.learned_listener import LearnedListenerSpymaster
from codenames.spymasters.pick_temperature_listener import tempered_gain_and_penalty

MODEL = "listener_gbt_fresh_split.txt"
# Picks 1..7 (7 = every later pick); cache/fresh_split_temps_plain_no_k_d0.6_fit_tune_on_plain_test.json.
PICK_TEMPERATURES = (1.026, 1.137, 1.372, 1.626, 1.863, 2.238, 2.904)


class FreshSplitListenerSpymaster(LearnedListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None,
                 max_number: int | None = None,
                 pick_temperatures: tuple[float, ...] | list[float] = PICK_TEMPERATURES, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir, model_path=Path(model_path or cache_dir / MODEL),
                         max_number=max_number, **kwargs)
        self.pick_temperatures = tuple(float(t) for t in pick_temperatures)

    def _gain_and_penalty(self, s_own, s_bad, costs, max_k, s_out, words=None):
        return tempered_gain_and_penalty(s_own, s_bad, costs, max_k, self.pick_temperatures, s_out)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return super().model_files({**params, "model_path": params.get("model_path") or cache_dir / MODEL})
