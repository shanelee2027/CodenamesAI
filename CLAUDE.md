# CodenamesAI

Design spec and source of truth: [`docs/SCOPE.md`](docs/SCOPE.md). Read it
before proposing any architecture change; if a change would contradict it,
say so explicitly and ask before diverging.

## Conventions (SCOPE.md §8)

- Python 3.11+, type hints on public functions, pytest for tests.
- No notebooks in VCS.
- One module at a time — do not build several at once on a fresh codebase.
- Commit at every milestone. Keep `docs/log.md` updated as work proceeds.

## Milestone checklist (SCOPE.md §9)

- [ ] M0 — Corpus collection (start immediately)
- [x] M1 — Board and legality
- [ ] M2 — GloVe and similarity tensor
- [ ] M3 — Inspector
- [ ] M4 — Remaining embedding spaces + fastText training
- [ ] M5 — Guesser pool
- [ ] M6 — Arena
- [ ] M7 — Features and data generation
- [ ] M8 — Scorer
- [ ] M9 — Evaluation and ablations
- [ ] M10 — Human evaluation
