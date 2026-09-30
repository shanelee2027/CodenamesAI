"""isa_listener's search with a listener that also reads ConceptNet.

**The change** (docs/versions/conceptnet_listener.md). On top of isa_listener's
three WordNet is-a features, seven from ConceptNet 5.7
(scripts/data/build_conceptnet_sims.py): its own is-a both ways, part-whole,
the number of specific relations, the strength of any direct edge, and
whether clue and word form a phrase either way round (Donald Duck, firefly).
Numberbatch carries ConceptNet only as a blurred similarity; these say which
relation holds, and whether the two words make a phrase at all.

The booster, `cache/listener_gbt_conceptnet.txt`, is the same recipe and the
same train split as isa_listener's, with all ten appended features. Search,
shortlist, reward and costs are learned_listener's.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

MODEL = "listener_gbt_conceptnet.txt"


class ConceptnetListenerSpymaster(LearnedListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None, **kwargs):
        model_path = Path(model_path) if model_path else Path(cache_dir) / MODEL
        super().__init__(*args, cache_dir=cache_dir, model_path=model_path, **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [Path(params.get("model_path") or cache_dir / MODEL),
                cache_dir / "isa_sims.npz", cache_dir / "conceptnet_sims.npz"]
