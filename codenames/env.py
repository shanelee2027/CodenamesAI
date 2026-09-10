"""Load `.env` from the project root into `os.environ`.

`ANTHROPIC_API_KEY` deliberately does not live in the shell profile.
Claude Code reads that same variable, and exporting it globally can move
Claude Code itself onto API billing instead of the subscription -- the
opposite of the intent, since the key exists to pay for *evaluation*
calls, not for editing this repo. Keeping it in a gitignored file that
only this package reads means the experiment can spend money and the
editor cannot.

Zero dependencies on purpose: python-dotenv would be a new entry in
pyproject for ~20 lines of parsing, and this file is read once per
process at the point an API client is constructed.

Existing environment variables always win, so `ANTHROPIC_API_KEY=... python
script.py` and CI secrets override the file rather than being silently
replaced by it.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

_loaded = False


def parse_env(text: str) -> dict[str, str]:
    """`KEY=value` per line. Blank lines and `#` comments are skipped, as
    is a leading `export `, so the file can also be `source`d from a shell
    when that's more convenient than going through Python. Surrounding
    single or double quotes are stripped; nothing else is interpreted --
    no variable expansion, no escapes, since an API key is an opaque
    string and guessing at shell semantics would only corrupt it."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Populate `os.environ` from `.env`; returns the names/values found.

    Idempotent for the default path -- callers can invoke it defensively
    at every entry point without re-reading the file. A missing file is
    not an error: most of this project never makes an API call, and
    importing a module shouldn't require a key to exist."""
    global _loaded
    target = DEFAULT_ENV_FILE if path is None else Path(path)
    if path is None and _loaded and not override:
        return {}
    try:
        text = target.read_text()
    except OSError:
        if path is None:
            _loaded = True
        return {}

    values = parse_env(text)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    if path is None:
        _loaded = True
    return values


def require(name: str) -> str:
    """Fetch a variable, loading `.env` first, with an error that says how
    to fix it rather than just naming the missing key."""
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Put it in {DEFAULT_ENV_FILE} as `{name}=...` "
            f"(see .env.example; the file is gitignored), or export it for one "
            f"command: `{name}=... python ...`"
        )
    return value
