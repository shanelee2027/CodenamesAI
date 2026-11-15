"""The GBT spymaster of the thesis: learned_listener's expected-reward search
with a listener trained from scratch on a three-way split of the online
game's 800 words (docs/versions/fresh_split_listener.md).

**The listener.** `cache/listener_gbt_fresh_split.txt`
(scripts/pipeline/train_fresh_split_listener.py, variant `no_k`): 72 features,
fitted only on gpt-oss rankings of positions generated on the split's 400
training words, with every z-score over the 800 (sidecar
`listener_gbt_fresh_split.meta.json`). The feature blocks were chosen by R² on
the 150 tuning words; the 250 test words are for the final evaluation only.
It does not read the clue's number: on the tuning words dropping `k` changed
R² by +0.0001 [-0.0008, +0.0011], so the predicted pick order is the same for
every number and one prediction per clue is exact.

**The search** is learned_listener's: the Gaussian expected-reward model
shortlists 200 clues, the listener rescores them, and each clue's number is
the one with the highest expected reward under the game's costs, computed
from the listener's pick order (codenames/pl_reward.py). No overpromise
charge, STOP model, win value or tie-break: the expected reward alone, so the
one difference from the Gaussian spymaster is the model of the guesser.
**No cap on the number** (`max_number=None`), as for the Gaussian spymaster
in the thesis.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

MODEL = "listener_gbt_fresh_split.txt"


class FreshSplitListenerSpymaster(LearnedListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None,
                 max_number: int | None = None, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir, model_path=Path(model_path or cache_dir / MODEL),
                         max_number=max_number, **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return super().model_files({**params, "model_path": params.get("model_path") or cache_dir / MODEL})
