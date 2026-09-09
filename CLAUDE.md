# CodenamesAI

Start with [`README.md`](README.md) for what this project is and how the
pieces fit together. Standing design rationale (why the feature vector,
guesser pool, and reward formula are built the way they are) lives in
[`docs/design-decisions.md`](docs/design-decisions.md) — read it before
proposing an architecture change, and if a change would contradict
something there, say so explicitly and ask before diverging.

The project is no longer organized around milestones. It's in an iteration
phase: the initial build is done, and work now proceeds as a sequence of
models, each trying to improve on the last. [`docs/versions/`](docs/versions/)
documents what changed in each and what's open for the next —
check it before starting new model work, and add a new doc rather
than silently changing what "the model" means out from under the current
one.

How that iteration is *supposed* to work — the spymaster registry, the
rollout cache, the frozen eval suite — is in
[`docs/iteration-architecture.md`](docs/iteration-architecture.md). Read it
before changing the training or evaluation pipeline.

## Naming models

**Models get descriptive names, not version numbers.** A name should say
what makes the model different (`k_cause_mlp`, not `v2`), because a number
carries no information about what changed and forces a lookup every time.
One file under `codenames/spymasters/`, one entry in
`configs/spymasters.json`, one doc under `docs/versions/` named for the
model, and nothing else edited.

`v1` and `v1.1` predate this convention and keep their names for now; they
will be renamed or removed rather than grandfathered permanently. Note
that they were also trained against the old 60-word holdout, so their
numbers are **not** comparable with models trained under the current
150-word split — see `docs/iteration-architecture.md` step 5.

## Cache layout

Generated artifacts under `cache/` (gitignored) are named for what they
are, not for the run that made them: `cache/rollouts/`,
`cache/training_data/`, `cache/checkpoints/`, `cache/llm_store.db`. Older
artifacts predating this (`cache/m9/`, `cache/arena_blend.db`,
`cache/sanity_check2.db`) are left as they are rather than renamed —
they're gitignored local data, and renaming them would break nothing but
prove nothing either.

`cache/llm_store.db` is the exception worth care: it holds every paid LLM
response, and since LLM output isn't deterministic even at temperature 0,
it is the only thing making past evaluations reproducible. Back it up.

## Conventions

- Python 3.11+, type hints on public functions, pytest for tests.
- One module at a time — do not build several at once on a fresh codebase.
- Commit at every meaningful step. Keep `docs/log.md` updated as work
  proceeds, recording what was expected and what actually happened —
  reconstructing the reasoning later is much harder than taking notes now.
- Before implementing anything non-obvious, describe two structural options
  and what breaks with each, then recommend one.
- Flag design choices the user might not notice. This is a graded project
  that will be defended orally; every file needs to be explicable by the
  author.
