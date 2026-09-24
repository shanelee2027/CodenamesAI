"""The learned listener retrained with association counts, and a "pass" priced
on the absolute level those counts give it.

**The listener.** `cache/listener_gbt_assoc_w0.3.txt`: the incumbent's recipe
plus a Poisson term on free-association counts (docs/log.md, "Association
counts"). The board softmax cannot tell a clue that points strongly at one word
(4, 2, 2) from one that points weakly (2, 0, 0); the counts are not normalised
over the board, so this model's scores carry a level that means something:
exp(a*s + b) is the rate at which the teacher names that word when it free-
associates from the clue, with (a, b) from `<model>.assoc.json` (the fit on
training rows). On unseen clues it ranks boards by that level at rho 0.51
against the incumbent's 0.41, at no cost on Sonnet's picks.

**The pass.** `pass_rate` adds one outside option with a FIXED score, the one a
word would need to be named in that fraction of association lists:

    s_out = (log(pass_rate) - b) / a

It enters the reward exactly as the decoy outside option does
(codenames/pl_reward.py): a non-team alternative with cost zero, so picking it
ends the turn and scores nothing. A sharp clue, whose target is named in most
lists, sits far above it and is barely touched; a vague clue, whose best word
is rarely named, loses much of its first pick to it and so its reward. That is
the 4,2,2-vs-2,0,0 distinction put to work. The decoy version priced the same
thing against "a random word under this clue", which moves with the clue and
so cannot see a clue-wide level; this one is a constant.

`pass_rate=None` switches the pass off and leaves only the retrained scores --
the control for whether the scores alone change anything.

Not swept. 0.05 is a starting point: on real positions 58% of boards have no
word named in any of five lists while named targets sit at 0.6-1.0, so 0.05 is
small against a sharp clue and large against a vague one. Against gpt-oss,
which never passes, any pass is expected to look bad in the arena (docs/log.md,
the outside_n sweep); the human study is where it can show.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

MODEL = "listener_gbt_assoc_w0.3.txt"
PASS_RATE = 0.05


def rate_link(model_path: Path) -> tuple[float, float]:
    """(a, b) with rate = exp(a*s + b), refitted on training rows."""
    link = json.loads(model_path.with_suffix(".assoc.json").read_text())["refit_on_train"]
    return float(link["a"]), float(link["b"])


class AssociationListenerSpymaster(LearnedListenerSpymaster):
    def __init__(self, pass_rate: float | None = PASS_RATE, *, cache_dir: Path = DEFAULT_CACHE_DIR,
                 model_path: Path | None = None, **kwargs):
        model_path = Path(model_path) if model_path else Path(cache_dir) / MODEL
        super().__init__(cache_dir=cache_dir, model_path=model_path, **kwargs)
        if kwargs.get("outside_n"):
            raise ValueError("the decoy outside option and the fixed pass are alternatives; use one")
        self.pass_rate = pass_rate
        self.a, self.b = rate_link(model_path)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        model = Path(params.get("model_path") or cache_dir / MODEL)
        return [model, model.with_suffix(".assoc.json")]

    def _fixed_outside_score(self) -> float | None:
        if self.pass_rate is None:
            return None
        return float((np.log(self.pass_rate) - self.b) / self.a)

    def rate(self, scores: np.ndarray) -> np.ndarray:
        """Association rate per list for listener scores: the level made
        readable, for display."""
        return np.exp(self.a * np.asarray(scores, dtype=np.float64) + self.b)
