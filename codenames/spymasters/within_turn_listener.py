"""conceptnet_listener with the guesser's later picks modelled as their own
choices (codenames/sequential_listener.py; docs/versions/within_turn_listener.md).

**The problem.** Every listener spymaster so far scores a clue once and, for
pick 2, 3 and 4, removes the words already taken and renormalises the same
scores (Plackett-Luce). So pick 2 cannot depend on what pick 1 was, and it is
exactly as sharp as pick 1. Measured, later picks are overconfident, and
pick-2 accuracy is 0.41 where the guesser's own repeated answers reach
0.68-0.75. The user's example is "uniform 4" for Cap, Fire, Orange and Glove,
where after Cap the listener gave Fire 24%.

**The change.** Only the reward. At pick j >= 2 the logits become

    alpha * s(w) + beta * max similarity of w to a word already picked,
    alpha = exp(a + b * (best score left - best score picked)),

with (a, b, beta) per pick, fitted by maximum likelihood on the stored
rankings' later picks (scripts/pipeline/train_sequential_listener.py, saved
in cache/sequential_listener.json). The fit gave beta ~5: a second word close
to the first is strongly preferred, the "Cap then Glove" pull. alpha < 1:
later picks are flatter. The reward is computed exactly over which own words
have been picked, as in pick_temperature_listener
(sequential_listener.sequential_gain_and_penalty). The listener, shortlist,
costs and search are conceptnet_listener's.

**Costs.** pick_temperature_listener lost partly because the costs (neutral
0.2, opponent 1, assassin 10) were tuned under the old, overconfident
reward. This model's doc records the costs it was run with; a cost sweep
under the new reward is part of evaluating it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codenames.listener_net import WordVectors
from codenames.sequential_listener import SequentialParams, pair_similarity, sequential_gain_and_penalty
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.conceptnet_listener import ConceptnetListenerSpymaster

PARAMS = "sequential_listener.json"
_VW: dict = {}


class WithinTurnListenerSpymaster(ConceptnetListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, sequential_path: Path | None = None, **kwargs):
        super().__init__(*args, cache_dir=cache_dir, **kwargs)
        path = Path(sequential_path) if sequential_path else Path(cache_dir) / PARAMS
        self.sequential = SequentialParams.from_dict(json.loads(path.read_text()))
        if "vw" not in _VW:
            _VW["vw"] = WordVectors()
        self._vw = _VW["vw"]

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [*super().model_files(params), Path(params.get("sequential_path") or cache_dir / PARAMS),
                cache_dir / "word_vectors.npz"]

    def _gain_and_penalty(self, s_own, s_bad, costs, max_k, s_out, words=None):
        if words is None:
            raise ValueError("within_turn_listener's reward needs the board words")
        sim = pair_similarity(words, self._vw)
        return sequential_gain_and_penalty(s_own, s_bad, costs, max_k, sim, self.sequential, s_out)
