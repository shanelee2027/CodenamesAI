"""Guesser pool registry (SCOPE.md §M5).

Pool composition lives in configs/guesser_pool.json, not in code, per
SCOPE.md §3: "Pool composition lives in a config file... Results are
reported as 'under pool configuration X, we observe Y.'" This module only
knows how to build guessers *from* a config; it has no opinion on what the
pool should contain.

Config format: {"guessers": [{"name", "type", "params", "held_out"}, ...]}.
`type` selects a class from GUESSER_CLASSES. `params` are passed as
keyword args to that class's constructor. A wrapper guesser (noisy,
confidence_threshold) references its base either:
  - by name (a string) -- that name must appear *earlier* in the list,
    since entries are built in order and a wrapper's base must already
    exist as a separately-visible pool member; or
  - inline (a nested `{"type", "params"}` object, no "name") -- built
    anonymously and never added to the pool, for when a base only exists
    to be wrapped and shouldn't also show up as its own pool entry (e.g.
    configs/guesser_pool.json's noisy_* entries, which wrap a bare
    single_space guesser that isn't meant to be independently sampled).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from codenames.guessers.base import Guesser
from codenames.guessers.blend import BlendGuesser
from codenames.guessers.confidence_threshold import ConfidenceThresholdGuesser
from codenames.guessers.history_aware import HistoryAwareGuesser
from codenames.guessers.llm import LLMGuesser
from codenames.guessers.noisy import NoisyGuesser
from codenames.guessers.rank_based import RankBasedGuesser
from codenames.guessers.single_space import SingleSpaceGuesser

DEFAULT_POOL_CONFIG = Path(__file__).parent.parent.parent / "configs" / "guesser_pool.json"

GUESSER_CLASSES: dict[str, type[Guesser]] = {
    "single_space": SingleSpaceGuesser,
    "blend": BlendGuesser,
    "rank_based": RankBasedGuesser,
    "noisy": NoisyGuesser,
    "confidence_threshold": ConfidenceThresholdGuesser,
    "history_aware": HistoryAwareGuesser,
    "llm": LLMGuesser,
}


@dataclass
class GuesserEntry:
    name: str
    guesser: Guesser
    held_out: bool


def _resolve_base(base_spec, entry_name: str, built: dict[str, Guesser]) -> Guesser:
    if isinstance(base_spec, str):
        if base_spec not in built:
            raise ValueError(
                f"guesser {entry_name!r} references base {base_spec!r}, "
                "which isn't defined earlier in the config"
            )
        return built[base_spec]
    if isinstance(base_spec, dict):
        # Anonymous inline base -- built but never added to `built`, so it
        # can't be referenced by name and never appears as its own pool entry.
        return _build_one(base_spec, built)
    raise ValueError(
        f"guesser {entry_name!r}'s base must be a string name or an inline "
        f"{{'type', 'params'}} object, got {base_spec!r}"
    )


def _build_one(entry_config: dict, built: dict[str, Guesser]) -> Guesser:
    guesser_type = entry_config["type"]
    if guesser_type not in GUESSER_CLASSES:
        raise ValueError(f"unknown guesser type {guesser_type!r}, must be one of {list(GUESSER_CLASSES)}")
    cls = GUESSER_CLASSES[guesser_type]
    params = dict(entry_config.get("params", {}))
    if "base" in params:
        entry_name = entry_config.get("name", "<anonymous>")
        params["base"] = _resolve_base(params["base"], entry_name, built)
    return cls(**params)


def load_pool(config: Path | dict = DEFAULT_POOL_CONFIG) -> dict[str, GuesserEntry]:
    """`config` is either a path to a pool config file, or an
    already-parsed config dict (e.g. a copy of one with `noise_std`
    overridden in memory -- see scripts/web_inspector.py's per-noise-level
    pools -- without needing to write a temp file first)."""
    parsed = json.loads(config.read_text()) if isinstance(config, Path) else config
    built: dict[str, Guesser] = {}
    entries: dict[str, GuesserEntry] = {}
    for entry_config in parsed["guessers"]:
        name = entry_config["name"]
        if name in built:
            raise ValueError(f"duplicate guesser name {name!r} in {config}")
        guesser = _build_one(entry_config, built)
        built[name] = guesser
        entries[name] = GuesserEntry(name=name, guesser=guesser, held_out=entry_config.get("held_out", False))
    return entries


def training_pool(config_path: Path = DEFAULT_POOL_CONFIG) -> dict[str, Guesser]:
    """Guessers training code is allowed to use. Never includes held-out
    guessers -- see SCOPE.md §3: "Training code must never touch them.\""""
    return {name: e.guesser for name, e in load_pool(config_path).items() if not e.held_out}


def held_out_pool(config_path: Path = DEFAULT_POOL_CONFIG) -> dict[str, Guesser]:
    """The evaluation-only guessers. Off-diagonal results against these
    are what actually matter per SCOPE.md §6's arena note."""
    return {name: e.guesser for name, e in load_pool(config_path).items() if e.held_out}
