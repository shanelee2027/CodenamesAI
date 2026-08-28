# CodenamesAI

Start with [`README.md`](README.md) for what this project is and how the
pieces fit together. Standing design rationale (why the feature vector,
guesser pool, and reward formula are built the way they are) lives in
[`docs/design-decisions.md`](docs/design-decisions.md) — read it before
proposing an architecture change, and if a change would contradict
something there, say so explicitly and ask before diverging.

The project is no longer organized around milestones. It's in an iteration
phase: the initial build is done, and work now proceeds as a sequence of
model versions, each trying to improve on the last. [`docs/versions/`](docs/versions/)
documents what changed in each version and what's open for the next one —
check it before starting new model work, and add a new version doc rather
than silently changing what "the model" means out from under the current
one.

## Conventions

- Python 3.11+, type hints on public functions, pytest for tests.
- No notebooks in VCS.
- One module at a time — do not build several at once on a fresh codebase.
- Commit at every meaningful step. Keep `docs/log.md` updated as work
  proceeds, recording what was expected and what actually happened —
  reconstructing the reasoning later is much harder than taking notes now.
- Before implementing anything non-obvious, describe two structural options
  and what breaks with each, then recommend one.
- Flag design choices the user might not notice. This is a graded project
  that will be defended orally; every file needs to be explicable by the
  author.
