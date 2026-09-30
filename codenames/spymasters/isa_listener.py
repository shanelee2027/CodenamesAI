"""learned_listener with a listener that knows "is a kind of".

**The problem** (docs/log.md, "Category clues: members against associates").
The incumbent's listener has no feature that is directional: Wu-Palmer
similarity and every embedding are symmetric, so for "fruit" it ranked Pie
above real fruit on average, for "planet" it gave Moon 14% of first picks, and
the associate halved the probability it gave "clue 3" of landing, where
gpt-oss put the three members first on 50 of 50 boards. With the associate as
the assassin the clue went negative and was never given.

**The change.** Three features from WordNet's hypernym tree
(scripts/data/build_isa_sims.py, codenames/listener_features.py): whether a
board word is a kind of the clue (isa), whether the clue is a kind of the word
(isa_rev), and how many board words are (isa_n), which lets the trees explain
the associate away when the members are all there. The booster,
`cache/listener_gbt_isa.txt`, is the incumbent's current training recipe with
them added, fitted on the train split and early-stopped on val exactly as the
table in docs/versions/isa_listener.md measured it. Nothing else differs from
learned_listener: the same shortlist, reward, costs and search.

**One difference that is not the features.** The deployed incumbent booster
(cache/listener_gbt.txt) is an older fit, 380 trees at learning rate 0.05; this
one is ~2,600 trees at 0.01, the recipe adopted since. The doc separates the
two effects with a same-recipe booster without the new features.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

MODEL = "listener_gbt_isa.txt"


class IsaListenerSpymaster(LearnedListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None, **kwargs):
        model_path = Path(model_path) if model_path else Path(cache_dir) / MODEL
        super().__init__(*args, cache_dir=cache_dir, model_path=model_path, **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [Path(params.get("model_path") or cache_dir / MODEL), cache_dir / "isa_sims.npz"]
