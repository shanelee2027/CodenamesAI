# CodenamesAI — agent guide

A non-LLM Codenames spymaster. Read [`README.md`](README.md) first for what the
project is, then [`docs/design-decisions.md`](docs/design-decisions.md) for the
standing rationale behind the feature vector, guesser pool and reward formula.

**[`CLAUDE.md`](CLAUDE.md) is the authoritative conventions file.** It is not
Claude-specific; this file exists because Codex looks for `AGENTS.md`, and it
repeats only what you are most likely to get wrong. Where the two disagree,
`CLAUDE.md` wins.

## Running anything

**Use `.venv/bin/python`, never bare `python3`.** The system interpreter has
none of the dependencies — `import openai` and `import lightgbm` both fail. There
is no `python` on PATH at all.

```bash
.venv/bin/python -m pytest -q          # ~370 tests, ~30s
.venv/bin/python scripts/pipeline/train_listener.py --help
```

Scripts that use multiprocessing must be run from a file, not a heredoc: a
spawned worker re-imports `__main__` and dies with `FileNotFoundError: '<stdin>'`.

## Things that fail in ways that do not name their cause

- **Guessers are spec strings**, e.g. `--guesser deepinfra:openai/gpt-oss-120b`
  or `anthropic:claude-sonnet-5:medium` (`codenames/guessers/registry.py::build_guesser`).
  A bare name like `noisy_glove` is a free synthetic guesser, not the LLM.
- **Arena workers are processes and each loads the similarity tensor and the
  listener** (~1.1 GB). Use processes x threads (`--max-workers 6
  --threads-per-worker 16`) rather than many processes; 48 processes triggered
  the OOM killer and took the editor down with it. Data collection is *threaded*, not
  processed, so it can go to 100 workers happily.
- **Never `pkill -f <script>`** — the pattern matches your own shell and kills
  it (exit 144). Kill explicit PIDs.
- **The teacher is gpt-oss, the evaluation guesser is Sonnet.** Never train on
  the evaluation guesser's rankings: that turns the evaluation into a self-test.

## Secrets

API keys live in a gitignored `.env` at the repo root (see `.env.example`),
loaded by `codenames/env.py` where a client is built. **Do not export them from
the shell profile**: coding agents read the same variables, and a global export
can move the agent itself onto API billing. Never print a key value.

Codex authenticates separately (`codex login`, or `OPENAI_API_KEY`) and does not
collide with `ANTHROPIC_API_KEY` / `DEEPINFRA_API_KEY`.

## Cache discipline

`cache/` is gitignored and artifacts are named for what they are, not for the
run that made them. **`cache/llm_store.db` is the one to be careful with**: it
holds every paid LLM response, and since LLM output is not deterministic even at
temperature 0, it is the only thing making past evaluations reproducible. Do not
delete or rewrite it. Never write a fabricated ranking into it — the guesser
raises rather than backfill, deliberately.

## Working conventions

- Python 3.11+, type hints on public functions, pytest.
- One module at a time.
- **Commit at every meaningful step, and keep `docs/log.md` updated** with what
  was expected and what actually happened. Reconstructing reasoning later is
  much harder than taking notes now. The log is also where negative results and
  retractions go — several numbers in it have been withdrawn on re-measurement,
  and that history is deliberate.
- Before implementing anything non-obvious, describe two structural options and
  what breaks with each, then recommend one.
- **Models get descriptive names, not version numbers** (`decoy_listener`, not
  `v2`): one file under `codenames/spymasters/`, one entry in
  `configs/spymasters.json`, one doc under `docs/versions/`.
- This is a graded project that will be defended orally. Every file needs to be
  explicable by the author, so flag design choices the author might not notice
  rather than burying them.
