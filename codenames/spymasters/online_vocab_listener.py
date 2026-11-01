"""floored_overpromise_listener with its two boosters retrained for the online
game's 800 words (docs/versions/online_vocab_listener.md).

**Why.** codenames.game deals from 800 words, not the base game's 400 that
every earlier booster was trained and normalised on. The listener transferred
to the new words without a gap (docs/log.md, 2026-10-30), so this is not a
fix; it is more data on the words the model is now dealt, and z-scores over
the population it is dealt from.

**What changed: only the two booster files.**
- The listener: `cache/listener_gbt_online_vocab.txt`, the incumbent's 67
  columns, trained on the incumbent's rows plus 9,980 generated positions on
  550 of the 800 words, with every z-scored feature over the 800 (its sidecar
  `listener_gbt_online_vocab.meta.json` says so, and the bundle loads the
  online tables for it).
- The STOP listener: `cache/listener_gbt_stop_online_vocab.txt`, the strict
  "guess" set, on the same rows and footing, with the new positions' labels.

Search, reward, V, the overpromise charge and its floor are
floored_overpromise_listener's.
"""

from __future__ import annotations

from pathlib import Path

from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.floored_overpromise_listener import FlooredOverpromiseListenerSpymaster

MODEL = "listener_gbt_online_vocab.txt"
STOP_MODEL = "listener_gbt_stop_online_vocab.txt"


class OnlineVocabListenerSpymaster(FlooredOverpromiseListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, model_path: Path | None = None,
                 stop_model_path: Path | None = None, **kwargs):
        cache_dir = Path(cache_dir)
        super().__init__(*args, cache_dir=cache_dir, model_path=Path(model_path or cache_dir / MODEL),
                         stop_model_path=Path(stop_model_path or cache_dir / STOP_MODEL), **kwargs)

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return super().model_files({**params, "model_path": params.get("model_path") or cache_dir / MODEL,
                                    "stop_model_path": params.get("stop_model_path") or cache_dir / STOP_MODEL})
