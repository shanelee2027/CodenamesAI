"""Spymaster registry (docs/iteration-architecture.md step 2), mirroring
codenames/guessers/registry.py.

Spymaster composition lives in configs/spymasters.json, not hardcoded in
every script that needs one -- same "composition lives in a config file"
stance codenames/guessers/registry.py already takes for the guesser pool
(see docs/design-decisions.md). This module only knows how to build a
spymaster spec *from* a config; it has no opinion on which spymasters
should exist.

Config format: {"spymasters": [{"name", "type", "params", "trained"}, ...]}.
`type` selects a class from SPYMASTER_CLASSES. `params` are passed as
keyword args to that class's constructor. `trained` marks whether this
entry expects a trained artifact (a checkpoint) to exist before it can be
built -- not every spymaster trains (`centroid` and `linear_scorer` don't,
and neither does the planned non-deep-learning baseline), so no code path
may assume every entry has one. `configs/spymasters.json` only lists the
baselines, which need no checkpoint; a `learned` entry's `params` would
need a real `checkpoint_path` (produced by a specific training run, not a
fixed config value) filled in by the caller -- see `spec()`'s `overrides`.

**Picklability.** `codenames/arena.py:188` constructs spymasters *inside*
spawned worker processes -- it needs a `(class, kwargs)` spec, not a live
instance, since a live instance (especially one holding a torch model)
isn't guaranteed to survive a `spawn`-context pickle/unpickle round trip.
`SpymasterEntry.spec` gives exactly that; `SpymasterEntry.build()` is for
callers (e.g. scripts/web_inspector.py) that want a live instance directly,
in the same process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from codenames.spymasters.base import Spymaster
from codenames.spymasters.centroid import CentroidSpymaster
from codenames.spymasters.learned import LearnedSpymaster
from codenames.spymasters.linear_scorer import LinearScorerSpymaster
from codenames.spymasters.oracle import OracleSpymaster
from codenames.spymasters.random_clue import RandomSpymaster

DEFAULT_SPYMASTER_CONFIG = Path(__file__).parent.parent.parent / "configs" / "spymasters.json"

SPYMASTER_CLASSES: dict[str, type[Spymaster]] = {
    "random": RandomSpymaster,
    "centroid": CentroidSpymaster,
    "linear_scorer": LinearScorerSpymaster,
    "oracle": OracleSpymaster,
    "learned": LearnedSpymaster,
}


@dataclass
class SpymasterEntry:
    name: str
    cls: type[Spymaster]
    params: dict
    trained: bool

    @property
    def spec(self) -> tuple[type[Spymaster], dict]:
        """Picklable (class, kwargs) form -- see this module's docstring
        and codenames/arena.py:188."""
        return (self.cls, self.params)

    def build(self) -> Spymaster:
        return self.cls(**self.params)


def _build_entry(entry_config: dict) -> SpymasterEntry:
    name = entry_config["name"]
    spymaster_type = entry_config["type"]
    if spymaster_type not in SPYMASTER_CLASSES:
        raise ValueError(f"unknown spymaster type {spymaster_type!r}, must be one of {list(SPYMASTER_CLASSES)}")
    return SpymasterEntry(
        name=name,
        cls=SPYMASTER_CLASSES[spymaster_type],
        params=dict(entry_config.get("params", {})),
        trained=entry_config.get("trained", False),
    )


def load_spymasters(config: Path | dict = DEFAULT_SPYMASTER_CONFIG) -> dict[str, SpymasterEntry]:
    """`config` is either a path to a spymaster config file, or an
    already-parsed config dict (same in-memory-override convenience
    codenames/guessers/registry.py::load_pool offers)."""
    parsed = json.loads(config.read_text()) if isinstance(config, Path) else config
    entries: dict[str, SpymasterEntry] = {}
    for entry_config in parsed["spymasters"]:
        entry = _build_entry(entry_config)
        if entry.name in entries:
            raise ValueError(f"duplicate spymaster name {entry.name!r} in {config}")
        entries[entry.name] = entry
    return entries


def spymaster_spec(
    name: str, config: Path | dict = DEFAULT_SPYMASTER_CONFIG, **overrides
) -> tuple[type[Spymaster], dict]:
    """One entry's (class, kwargs) spec, with `overrides` merged into its
    config params -- e.g. a `learned` entry's `checkpoint_path`, which
    only a training run (not this static config) can supply. Raises
    KeyError if `name` isn't in `config`."""
    entries = load_spymasters(config)
    if name not in entries:
        raise KeyError(f"unknown spymaster {name!r}, must be one of {list(entries)}")
    entry = entries[name]
    return entry.cls, {**entry.params, **overrides}
