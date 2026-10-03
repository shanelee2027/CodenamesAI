"""assoc_feature_listener with a listener that also reads the clue's
association profile and the reverse associations from each board word
(docs/versions/assoc_profile_listener.md).

**The change.** Eleven listener features from gpt-oss's free associations
(codenames/assoc_profile.py, table cache/assoc_profile.npz):
- the clue's profile, the same for every board word: how vague its
  association lists are (spread, overlap, agreement on the first word), and
  its rarity, concreteness, familiarity, frequency and WordNet senses;
- the reverse direction: whether each board word's own lists name the clue,
  how early, and that word's rank on the board.

No new gpt-oss calls are needed in play: the clue pool's lists and all 400
board words' lists are already owned.

The booster, `cache/listener_gbt_assoc_profile.txt`
(scripts/pipeline/train_profile_listener.py), is the assoc booster's recipe,
rows and split with the eleven features added. Search, shortlist, reward and
costs are learned_listener's. Under the win-probability objective it is used
as `win_prob_listener` with `model_path` set to this booster.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.assoc_feature_listener import AssocFeatureListenerSpymaster

MODEL = "listener_gbt_assoc_profile.txt"


class AssocProfileListenerSpymaster(AssocFeatureListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None, **kwargs):
        model_path = Path(model_path) if model_path else Path(cache_dir) / MODEL
        super().__init__(*args, cache_dir=cache_dir, model_path=model_path, **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [*super().model_files({**params, "model_path": params.get("model_path") or cache_dir / MODEL}),
                cache_dir / "assoc_profile.npz"]
