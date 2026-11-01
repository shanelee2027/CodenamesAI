"""learned_listener.bundle_normalization: a booster says which board words its
z-scored features were normalised over through a sidecar, and one without a
sidecar is on the base footing, as every booster before the online split."""

import json

from codenames.spymasters.learned_listener import bundle_normalization


def test_no_sidecar_is_base(tmp_path):
    assert bundle_normalization(tmp_path / "listener_gbt_x.txt") == "base"


def test_sidecar_names_the_footing(tmp_path):
    (tmp_path / "listener_gbt_x.meta.json").write_text(json.dumps({"normalization": "online"}))
    assert bundle_normalization(tmp_path / "listener_gbt_x.txt") == "online"


def test_sidecar_without_the_key_is_base(tmp_path):
    (tmp_path / "listener_gbt_x.meta.json").write_text("{}")
    assert bundle_normalization(tmp_path / "listener_gbt_x.txt") == "base"
