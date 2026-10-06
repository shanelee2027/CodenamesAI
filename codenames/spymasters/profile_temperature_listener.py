"""assoc_profile_listener with per-pick temperatures
(docs/versions/profile_temperature_listener.md).

The assoc profile booster is the most accurate listener that scores each
word once (R² 0.331 on the Sonnet 5.5 generated held-out set), and like every
such listener it is overconfident at later picks: it reuses pick 1's scores
for picks 2-4. Per-pick temperatures fix most of that at no cost in speed,
where pick-index boosters need a booster run per pick. At pick j the guesser
picks from the words left by softmax(s / tau_j), with pick 1 left at 1 and
tau_2..4+ fitted for this booster on gpt-oss val
(scripts/pipeline/train_sequential_listener.py --arm temperature): 1.09,
1.26, 1.41. The reward is pick_temperature_listener's exact one (expected own
words minus role costs, +1 / -0.2 / -1 / -10); everything else is
assoc_profile_listener's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.assoc_profile_listener import AssocProfileListenerSpymaster
from codenames.spymasters.pick_temperature_listener import tempered_gain_and_penalty

TEMPERATURES = "sequential_listener_assoc_profile_temperature.json"


def temperatures_from(path: Path) -> tuple[float, ...]:
    """(1, tau_2, tau_3, tau_4+) from a temperature-arm fit: its a_j scale
    pick j + 1's scores by exp(a_j), so tau = exp(-a_j)."""
    d = json.loads(Path(path).read_text())
    if any(d[k] and any(d[k]) for k in ("b", "beta", "gamma")):
        raise ValueError(f"{path} is not a temperature-only fit")
    return (1.0, *(float(np.exp(-a)) for a in d["a"]))


class ProfileTemperatureListenerSpymaster(AssocProfileListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, temperature_path: Path | None = None, **kwargs):
        super().__init__(*args, cache_dir=cache_dir, **kwargs)
        self.pick_temperatures = temperatures_from(Path(temperature_path or Path(cache_dir) / TEMPERATURES))

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [*super().model_files(params), Path(params.get("temperature_path") or cache_dir / TEMPERATURES)]

    def _gain_and_penalty(self, s_own, s_bad, costs, max_k, s_out, words=None):
        return tempered_gain_and_penalty(s_own, s_bad, costs, max_k, self.pick_temperatures, s_out)
