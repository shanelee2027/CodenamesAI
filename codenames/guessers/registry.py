"""Guesser pool registry.

Pool composition lives in configs/guesser_pool.json, not in code, per
docs/design-decisions.md's "composition lives in a config file, not in
code" note -- results are reported as "under pool configuration X, we
observe Y."
This module only
knows how to build guessers *from* a config; it has no opinion on what the
pool should contain.

Config format: {"guessers": [{"name", "type", "params", "held_out"}, ...]}.
`type` selects a class from GUESSER_CLASSES. `params` are passed as
keyword args to that class's constructor. A wrapper guesser (noisy)
references its base either:
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
from codenames.guessers.llm import LLMGuesser
from codenames.guessers.noisy import NoisyGuesser
from codenames.guessers.openai_compat import PROVIDERS, OpenAICompatGuesser
from codenames.guessers.single_space import SingleSpaceGuesser
from codenames.llm_store import DEFAULT_DB_PATH

DEFAULT_POOL_CONFIG = Path(__file__).parent.parent.parent / "configs" / "guesser_pool.json"

GUESSER_CLASSES: dict[str, type[Guesser]] = {
    "single_space": SingleSpaceGuesser,
    "noisy": NoisyGuesser,
    "llm": LLMGuesser,
    "openai_compat": OpenAICompatGuesser,
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
    overridden in memory for a sweep, without needing to write a temp
    file first)."""
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
    guessers -- training code must never touch them."""
    return {name: e.guesser for name, e in load_pool(config_path).items() if not e.held_out}


def held_out_pool(config_path: Path = DEFAULT_POOL_CONFIG) -> dict[str, Guesser]:
    """The evaluation-only guessers of a pool, if it marks any."""
    return {name: e.guesser for name, e in load_pool(config_path).items() if e.held_out}


def build_guesser(spec: str, pool_config: Path | dict = DEFAULT_POOL_CONFIG) -> Guesser:
    """One guesser from a one-line spec, or by name from `pool_config`.

        anthropic:<model>[:<effort>]    LLMGuesser, e.g. anthropic:claude-sonnet-5:medium
        <provider>:<model>[:<effort>][:temperature=<t>][:stop=1]
                                        OpenAICompatGuesser, for any provider in
                                        codenames/guessers/openai_compat.py::PROVIDERS,
                                        e.g. deepinfra:openai/gpt-oss-120b:low:temperature=1.0
        <name>                          an entry in `pool_config` (the synthetic guessers)

    Every LLM guesser caches to cache/llm_store.db, keyed by model and
    effort (and temperature, when set), so two specs never share an answer and
    the same spec never pays twice.

    Effort is part of the cache key, and its default differs by provider on
    purpose. An Anthropic model left without one runs with thinking disabled
    (see LLMGuesser.__init__). gpt-oss-120b left to its own default reasons at
    medium: 8x the completion tokens for the same one-shot judgment, and on a
    small budget an empty answer -- so OpenAICompatGuesser defaults to low.

    Temperature is accepted for OpenAI-compatible providers only: Claude Sonnet
    5 and later reject the sampling parameters outright. `stop=1` lets the
    guesser end its turn early (OpenAICompatGuesser.allow_stop), likewise
    OpenAI-compatible only.
    """
    provider, sep, rest = spec.partition(":")
    if not sep:
        return load_pool(pool_config)[spec].guesser
    model, _, tail = rest.partition(":")
    if not model:
        raise ValueError(f"guesser spec {spec!r} names no model")
    fields = tail.split(":") if tail else []
    effort = fields.pop(0) if fields and "=" not in fields[0] else ""
    options = {}
    for f in fields:
        key, eq, value = f.partition("=")
        if not eq or key not in ("temperature", "stop"):
            raise ValueError(f"unknown guesser option {f!r} in {spec!r}; "
                             "supported: temperature=<t>, stop=1")
        options[key if key == "temperature" else "allow_stop"] = (
            float(value) if key == "temperature" else value.lower() in ("1", "true", "yes"))
    if provider == "anthropic":
        if options:
            raise ValueError(f"{spec!r}: Anthropic guessers take no options here (temperature is "
                             "rejected by the API; stop is not implemented for them)")
        return LLMGuesser(model=model, effort=effort or None, cache_path=DEFAULT_DB_PATH)
    if provider in PROVIDERS:
        kwargs = {"reasoning_effort": effort} if effort else {}
        return OpenAICompatGuesser(model=model, provider=provider, cache_path=DEFAULT_DB_PATH,
                                   **kwargs, **options)
    raise ValueError(f"unknown guesser provider {provider!r} in {spec!r}; "
                     f"use anthropic or one of {sorted(PROVIDERS)}")
