"""conceptnet_listener's search with a listener that also reads gpt-oss's
free associations to the clue.

**The change** (docs/versions/assoc_feature_listener.md). Two features from
five free-association lists per clue (the clue shown alone, no board;
scripts/data/collect_associations.py --pool, scripts/data/build_assoc_sims.py):
- `assoc_share`: the share of the lists that name the board word;
- `assoc_rank`: the mean over lists of 1 / its position in the list.

The lists were bought for the spymaster's whole clue pool (rarity <= 10), so
the features exist for every clue it can give, not only for the ones in the
training rankings.

**Not to be confused with association_listener**, which used the same kind
of lists as a second training TARGET. Here they are INPUTS to the listener.

The booster, `cache/listener_gbt_assoc_features.txt`, uses the same recipe
and train split as conceptnet_listener's, with the two features added.
Search, shortlist, reward and costs are learned_listener's.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.conceptnet_listener import ConceptnetListenerSpymaster

MODEL = "listener_gbt_assoc_features.txt"


class AssocFeatureListenerSpymaster(ConceptnetListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None, **kwargs):
        model_path = Path(model_path) if model_path else Path(cache_dir) / MODEL
        super().__init__(*args, cache_dir=cache_dir, model_path=model_path, **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [Path(params.get("model_path") or cache_dir / MODEL),
                cache_dir / "isa_sims.npz", cache_dir / "conceptnet_sims.npz", cache_dir / "assoc_sims.npz"]
