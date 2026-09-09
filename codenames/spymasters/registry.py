"""Spymaster registry (docs/iteration-architecture.md step 2), mirroring
codenames/guessers/registry.py.

Spymaster composition lives in configs/spymasters.json, not hardcoded in
every script that needs one -- same "composition lives in a config file"
stance codenames/guessers/registry.py already takes for the guesser pool
(see docs/design-decisions.md). This module only knows how to build a
spymaster spec *from* a config; it has no opinion on which spymasters
should exist.

Config format: {"spymasters": [{"name", "type", "params", "trained",
"roles"}, ...]}.
`type` selects a class from SPYMASTER_CLASSES. `params` are passed as
keyword args to that class's constructor. `trained` marks whether this
entry expects a trained artifact (a checkpoint) to exist before it can be
built -- not every spymaster trains (`centroid` and `linear_scorer` don't,
and neither does the planned non-deep-learning baseline), so no code path
may assume every entry has one. `configs/spymasters.json` only lists the
baselines, which need no checkpoint; a `learned` entry's `params` would
need a real `checkpoint_path` (produced by a specific training run, not a
fixed config value) filled in by the caller -- see `spec()`'s `overrides`.

`roles` is how a script says *which kind* of spymaster it wants without
naming names: `scripts/pipeline/run_arena.py` takes the "baseline" role, and
`scripts/pipeline/run_two_team_arena.py` additionally takes "exploration" (its
`oracle` entry -- an upper-bound exploration tool, not a realistic
baseline, per the README). Adding a spymaster is then a config entry and
nothing else: it appears everywhere its roles say it belongs, with no
script edited. `roles` defaults to ("baseline",) precisely so a new entry
shows up by default rather than being silently invisible.

**Picklability.** `codenames/arena.py:188` constructs spymasters *inside*
spawned worker processes -- it needs a `(class, kwargs)` spec, not a live
instance, since a live instance (especially one holding a torch model)
isn't guaranteed to survive a `spawn`-context pickle/unpickle round trip.
`SpymasterEntry.spec` gives exactly that; `SpymasterEntry.build()` is for
callers (e.g. scripts/tools/web_inspector.py) that want a live instance directly,
in the same process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from codenames.spymasters.base import Spymaster
from codenames.spymasters.centroid import CentroidSpymaster
from codenames.spymasters.expected_words import ExpectedWordsSpymaster
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
    "expected_words": ExpectedWordsSpymaster,
}


@dataclass
class SpymasterEntry:
    name: str
    cls: type[Spymaster]
    params: dict
    trained: bool
    roles: tuple[str, ...]

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
        roles=tuple(entry_config.get("roles", ("baseline",))),
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


def spymaster_names(*roles: str, config: Path | dict = DEFAULT_SPYMASTER_CONFIG) -> list[str]:
    """Names of every entry carrying at least one of `roles`, in config
    order -- what a script asks for instead of hardcoding a name list, so
    a newly-added config entry needs no script change to show up. See
    this module's docstring on `roles`."""
    wanted = set(roles)
    return [name for name, entry in load_spymasters(config).items() if wanted & set(entry.roles)]
