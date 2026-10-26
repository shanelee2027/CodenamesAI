"""win_prob_overpromise_listener on a V table without ties
(docs/versions/margin_value_overpromise_listener.md).

**Why.** win_prob_overpromise_listener converts its cost per unattempted
promised word into win probability with `WinValue.one_word(a, b)`, read off
cache/win_value.npz. That table is made monotone by isotonic regression,
which pools neighbouring cells into exact ties, and at the opening
V(8, 9) = V(8, 8) = 0.637, so one_word(9, 8) = 0: on the starting team's
first clue the cost vanishes and finding a first word looks worth nothing.

**What changed: only V.** cache/win_value_margin.npz is built from the same
10,266 games as cache/win_value.npz (`--max-game-id 15907`) with
`--margin 0.5`: every step of V is at least half the step of the smooth
logistic fit it is shrunk toward, so no two cells tie (scripts/data/
build_win_value.py). The objective, the cost, the strict STOP listener and
the number are win_prob_overpromise_listener's. `value_path` overrides.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.win_prob_overpromise_listener import WinProbOverpromiseListenerSpymaster

MARGIN_VALUE = "win_value_margin.npz"


class MarginValueOverpromiseListenerSpymaster(WinProbOverpromiseListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, value_path: Path | None = None, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir, value_path=Path(value_path or cache_dir / MARGIN_VALUE), **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return super().model_files({**params, "value_path": params.get("value_path") or cache_dir / MARGIN_VALUE})
