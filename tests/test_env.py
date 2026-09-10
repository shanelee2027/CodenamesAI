"""codenames/env.py -- .env loading."""

import os

from codenames.env import load_env, parse_env, require

import pytest


def test_parses_plain_assignments():
    assert parse_env("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_skips_blanks_and_comments():
    assert parse_env("\n# note\nA=1\n\n  # indented\nB=2\n") == {"A": "1", "B": "2"}


def test_strips_export_prefix_so_the_file_can_also_be_sourced():
    assert parse_env("export A=1\n") == {"A": "1"}


def test_strips_surrounding_quotes():
    assert parse_env("A='1'\nB=\"2\"\n") == {"A": "1", "B": "2"}


def test_keeps_inner_characters_verbatim():
    """An API key is opaque -- no expansion, no escape handling, and '='
    inside the value survives."""
    assert parse_env("K=sk-ant-a=b$c#d\n") == {"K": "sk-ant-a=b$c#d"}


def test_ignores_lines_without_an_equals_sign():
    assert parse_env("nonsense\nA=1\n") == {"A": "1"}


def test_load_env_sets_missing_variables(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("PROBE_TEST_KEY=abc\n")
    monkeypatch.delenv("PROBE_TEST_KEY", raising=False)
    load_env(f)
    assert os.environ["PROBE_TEST_KEY"] == "abc"


def test_existing_environment_wins(tmp_path, monkeypatch):
    """A one-off `VAR=... python ...` must not be silently replaced by the
    file, or overriding for a single run would be impossible."""
    f = tmp_path / ".env"
    f.write_text("PROBE_TEST_KEY=from_file\n")
    monkeypatch.setenv("PROBE_TEST_KEY", "from_shell")
    load_env(f)
    assert os.environ["PROBE_TEST_KEY"] == "from_shell"


def test_override_forces_the_file_to_win(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("PROBE_TEST_KEY=from_file\n")
    monkeypatch.setenv("PROBE_TEST_KEY", "from_shell")
    load_env(f, override=True)
    assert os.environ["PROBE_TEST_KEY"] == "from_file"


def test_missing_file_is_not_an_error(tmp_path):
    """Most of this project never makes an API call; importing a module
    must not require a key to exist."""
    assert load_env(tmp_path / "nope.env") == {}


def test_require_error_says_how_to_fix_it(monkeypatch):
    monkeypatch.delenv("PROBE_TEST_MISSING", raising=False)
    with pytest.raises(RuntimeError, match=r"\.env"):
        require("PROBE_TEST_MISSING")
