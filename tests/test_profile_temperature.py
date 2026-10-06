"""profile_temperature_listener reads its temperatures from a temperature-arm fit."""

import json

import numpy as np
import pytest

from codenames.spymasters.profile_temperature_listener import temperatures_from


def test_temperatures_are_exp_minus_a_with_pick_one_at_one(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"a": [-0.1, -0.2, -0.3], "b": [0, 0, 0], "beta": [0, 0, 0], "gamma": [0, 0, 0]}))
    assert temperatures_from(f) == pytest.approx((1.0, *np.exp([0.1, 0.2, 0.3])))


def test_refuses_a_fit_with_more_than_temperatures(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"a": [0, 0, 0], "b": [0, 0, 0], "beta": [0.5, 0, 0], "gamma": [0, 0, 0]}))
    with pytest.raises(ValueError):
        temperatures_from(f)


def test_builds_from_the_registry_with_its_own_files():
    from codenames.spymasters.registry import spymaster_spec

    cls, kw = spymaster_spec("profile_temperature_listener")
    names = {p.name for p in cls.model_files(kw)}
    assert {"listener_gbt_assoc_profile.txt", "assoc_profile.npz",
            "sequential_listener_assoc_profile_temperature.json"} <= names
