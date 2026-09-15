# Iteration architecture

How a new spymaster gets built, trained, and evaluated. This is standing
rationale like [`design-decisions.md`](design-decisions.md), but about the
*workflow* rather than the model: what has to be true for "I have an idea
for a spymaster" to become "I have a number for it" without hand-editing
unrelated files or re-spending money.

Nothing here contradicts `design-decisions.md`; the registry and frozen
eval suite extend its "composition lives in a config file, results are
reported as *under configuration X*" stance to spymasters and to
evaluation.

## The goal

1. A new spymaster is one new file plus one config entry. No other file
   changes to make it runnable, trainable, or evaluable.
2. Whatever is expensive and model-independent is computed once and
   cached: embedding similarities (the similarity tensor), per-clue
   statistics (`clue_stats.npz`), and LLM responses (`llm_store.py`).
3. Evaluation against the LLM guesser is paid for once per model.

## Two roles for guessers

Settled, and it drives everything below:

- **Evaluation uses only LLM guessers.** Standardized across models, so
  results are comparable. The LLM model id is part of the eval suite's
  identity -- switching Haiku to Sonnet to Opus invalidates every cached
  response and makes old numbers non-comparable.
- **Every other guesser exists only for training.** Training against the
  LLM would overfit the spymaster to how one specific model thinks, and
  would collapse the train/eval distinction entirely.

`scripts/pipeline/run_arena.py`'s spymaster x guesser matrix is therefore a
training diagnostic, not a scoreboard. Its numbers must never be
presented as evaluation results.

## Step 1: TurnContext and a top-k interface

Two changes to `spymasters/base.py`, made together because both touch
every spymaster and every caller.

**A context object instead of loose arguments.** `give_clue(board,
sims)` had no turn counter, so a model wanting one had to reconstruct it
as `len(board.revealed)`. Models are board-state-only for now, but the
input set is expected to grow (clue history, past guesses). A dataclass absorbs that growth
without touching models that ignore the new field:

```python
@dataclass(frozen=True)
class TurnContext:
    board: Board | OpponentBoardView
    turn_index: int
```

**Top-k as the primary method.** A model that ranks the whole clue
vocabulary already produces `(clue, number, score)` triples; making that
the interface and `give_clue` a thin wrapper means one scoring path per
model instead of two that can drift.

```python
class Spymaster(ABC):
    @abstractmethod
    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor,
                  k: int) -> list[tuple[str, int, float]]: ...

    def give_clue(self, ctx, sims) -> tuple[str, int]:
        clue, number, _ = self.top_clues(ctx, sims, 1)[0]
        return clue, number
```

Reward parameters live on the model, so the arena never reads them. **Legality filtering
stays in `clue_search`** -- it is a rule of Codenames, identical for
every model, and must not be reimplemented per model.

## Step 2: A spymaster registry

Every runner used to keep its own private list of which spymasters
exist (`run_arena.py`, `run_two_team_arena.py`, and the since-removed
inspectors), so adding a model meant editing all of them.

`spymasters/registry.py` mirrors `guessers/registry.py`: a
`SPYMASTER_CLASSES` dict, a `configs/spymasters.json` giving each entry a
`name`, `type`, and `params`, and loaders that every script reads.

**Constraint:** `arena.py:188` constructs spymasters *inside* spawned
worker processes, so the registry must be able to hand out a picklable
`(class, kwargs)` spec, not just a live instance. A name-plus-params
config satisfies this more cleanly than a class reference does.

A model also declares whether it has a trained artifact. Not every
spymaster trains -- `centroid` and `linear_scorer` don't, and neither
does the planned non-deep-learning baseline -- so no code path may assume
a checkpoint exists.

## Step 3: A batched-scoring protocol

The GPU arenas used to be hardcoded to one model: they imported it
directly, reached into its torch internals, and computed expected reward
themselves. Any new model that scores the whole vocabulary then gets
either the slow per-process path or a forked copy of a subtle lockstep
loop.

The batched loop does five things inline. Only the first and last belong
to the arena:

| | who owns it |
|---|---|
| gather board views for active games | arena |
| build features | **model** |
| forward pass | **model** |
| expected reward -> `(best_n, scores)` | **model** |
| pick the best *legal* clue | arena (`clue_search`) |

So the boundary is: hand the spymaster board views, get back per-view
scores over the whole clue vocabulary.

```python
class BatchScoringSpymaster(Protocol):
    def score_batch(self, sims, contexts: list[TurnContext],
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """(best_n, scores) per context, indexed by sims.clue_words."""

    def to_device(self, device) -> None: ...
```

`to_device` replaces the arena reaching into a model's internals. The
single-board path routes through `score_batch` with a one-element list,
so there is exactly one scoring implementation per model.
`ExpectedWordsSpymaster` implements this protocol; `to_device` is a
no-op for it, since it scores in numpy on the CPU.

## Step 4: Cache rollouts, not features (retired)

This step existed to serve supervised training: rollouts
(`(board, clue, guesser, turn_index) -> (k, cause)`) were cached so a new
feature design cost a recompute rather than a re-simulation, measured at
~35x cheaper. It was built, measured, and removed along with the training
pipeline it fed -- see `docs/log.md` for the numbers.

The heading is kept so that references to steps 5-7 elsewhere in the
codebase stay correct. If a future model needs fitting, note that this
schema stored one turn's outcome, not a trajectory, so a value function
for multi-turn play would need a different one.

## Step 5: A frozen eval suite

- **Fixed board seeds.** Without them a rerun samples different boards
  and cannot be a cache hit, and two models' numbers aren't comparable.
- **Held-out board words, raised from 60 to 150 of 400.** Two random
  25-word boards drawn from a pool of `H` share about `625/H` words. At
  H=60 that is ~10.4 of 25 -- any two "independent" eval games share 40%
  of their board, so 300 games are nowhere near 300 independent samples.
  H=150 gives ~4.2. Going further buys little (H=200 gives ~3.1) while
  cutting training vocabulary below 250. Cost: training board vocabulary
  drops 340 -> 250. The mechanism is free (`board.py:208` is a set
  difference; the similarity tensor covers all 400 regardless).
- **The LLM model id is part of the suite identity.**

**Contamination warning.** `v1` and `v1.1` were trained with only 60
words held out, so they trained on 90 words that are now in the eval set.
They must not appear in the same table as a model trained under the new
split without being retrained, or the comparison flatters them.

## Step 6: An eval store

The prompt cache (`llm_store.py`) prevents re-paying for an identical
prompt. It does not prevent re-running an eval. Store results per game,
keyed by `(spymaster_id, suite_id, board_seed)`:

- a rerun costs nothing;
- going from 100 to 300 boards pays only for the 200 new games;
- a crashed run resumes.

`GameRecordStore` already writes one row per game and needs the right
key, not a rewrite.

**Identify a model by its content, not by a name.** A model's
`spymaster_id` has to change whenever anything that changes its clues
changes -- its parameters included -- or a cache keyed on it would
silently serve one model's expensive results as another's.

Because the whole game replays through cached responses, and because the
cache key does not mention the spymaster, two models that give the same
clue on the same remaining words get the *identical* LLM response. That
makes cross-model comparison paired rather than independently sampled --
a methodological benefit, not just a cost saving.

## Step 7: Naming and layout

**Done.** `scripts/` is grouped by *when you run it* — `data/` (build
time, once), `pipeline/` (the iteration loop), `tools/` (interactive) —
with [`scripts/README.md`](../scripts/README.md) as the index. The
grouping is constrained by the fact that scripts import their siblings
through `sys.path`, so anything that imports another script lives in the
same group; that held for all four such pairs without contortion.

Naming conventions for models and for `cache/` artifacts moved into
[`CLAUDE.md`](../CLAUDE.md), since that is what gets read at the start of
a session where a new model is about to be named.

**Existing `cache/` artifacts were deliberately not renamed.** `cache/m9/`,
`cache/arena_blend.db` and friends are gitignored local data; renaming
them would break nothing and prove nothing, while risking the one file
that genuinely matters (`cache/llm_store.db`). The convention applies to
what gets created from here on.

Historical entries in `docs/log.md` still reference the pre-move script
paths. That is intentional: the log is a record of what was true when
each entry was written, not a current-facing index. Every other document
was updated.

## Operational note

`cache/llm_store.db` is the record of every dollar spent, and since LLMs
are not deterministic even at temperature 0, it is also the only thing
making past evals reproducible. It lives in a gitignored directory with
no backup. It needs a real backup story.
