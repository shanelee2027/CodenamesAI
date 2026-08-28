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
- [x] M2 — GloVe and similarity tensor
- [x] M3 — Inspector
- [ ] M4 — Remaining embedding spaces + fastText training
- [x] M5 — Guesser pool
- [x] M6 — Arena
- [x] M7 — Features and data generation
- [x] M8 — Scorer
- [x] M9 — Evaluation and ablations
- [ ] M10 — Human evaluation
