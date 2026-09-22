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
.venv/bin/python -m pytest -q          # 383 tests, ~35s
.venv/bin/python scripts/pipeline/train_listener.py --help
```

Scripts that use multiprocessing must be run from a file, not a heredoc: a
spawned worker re-imports `__main__` and dies with `FileNotFoundError: '<stdin>'`.

## Things that fail in ways that do not name their cause

- **The LLM guesser is not in the default pool.** `configs/guesser_pool.json`
  holds only the synthetic guessers. Anything wanting the real teacher needs
  `--guesser-pool-config configs/guesser_pool_oss120b.json`. Omitting it used to
  surface as `BrokenProcessPool` with no cause; there is a guard now.
- **Arena workers are processes and each loads the similarity tensor** (~1.5 GB).
  Keep `--max-workers` at 10–12 on a 30 GB machine; 48 triggered the OOM killer
  and took the editor down with it. Data collection is *threaded*, not
  processed, so it can go to 100 workers happily.
- **Never `pkill -f <script>`** — the pattern matches your own shell and kills
  it (exit 144). Kill explicit PIDs.
- **The teacher is gpt-oss, not Sonnet.** `DEFAULT_MODEL` in
  `train_listener.py` still says `claude-sonnet-5`, but the store holds 62k
  gpt-oss responses against 10k Sonnet and the arena guesser is gpt-oss. Pass
  `--model deepinfra/openai/gpt-oss-120b+effort=low` explicitly.

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
- **Models get descriptive names, not version numbers** (`k_cause_mlp`, not
  `v2`): one file under `codenames/spymasters/`, one entry in
  `configs/spymasters.json`, one doc under `docs/versions/`.
- This is a graded project that will be defended orally. Every file needs to be
  explicable by the author, so flag design choices the author might not notice
  rather than burying them.
