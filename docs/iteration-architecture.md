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
   cached. Three such things exist: embedding similarities (done -- the
   similarity tensor), LLM responses (done -- `llm_store.py`), and
   guesser rollouts (missing -- see step 4).
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

`scripts/run_arena.py`'s spymaster x guesser matrix is therefore a
training diagnostic, not a scoreboard. Its numbers must never be
presented as evaluation results.

## Step 1: TurnContext and a top-k interface

Two changes to `spymasters/base.py`, made together because both touch
every spymaster and every caller.

**A context object instead of loose arguments.** Today `give_clue(board,
sims)` has no turn counter, so `LearnedSpymaster` reconstructs one as
`len(board.revealed)` -- its own docstring flags this as train/serve skew
risk. Models are board-state-only for now, but the input set is expected
to grow (clue history, past guesses). A dataclass absorbs that growth
without touching models that ignore the new field:

```python
@dataclass(frozen=True)
class TurnContext:
    board: Board | OpponentBoardView
    turn_index: int
```

**Top-k as the primary method.** `LearnedSpymaster.top_k_clues` already
returns `(clue, number, score)` triples; making that the interface and
`give_clue` a thin wrapper means one scoring path per model instead of
two that can drift.

```python
class Spymaster(ABC):
    @abstractmethod
    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor,
                  k: int) -> list[tuple[str, int, float]]: ...

    def give_clue(self, ctx, sims) -> tuple[str, int]:
        clue, number, _ = self.top_clues(ctx, sims, 1)[0]
        return clue, number
```

Reward parameters live on the model (already true of
`LearnedSpymaster`), so the arena never reads them. **Legality filtering
stays in `clue_search`** -- it is a rule of Codenames, identical for
every model, and must not be reimplemented per model.

## Step 2: A spymaster registry

Today three files keep their own private list of which spymasters exist
(`run_arena.py`, `run_two_team_arena.py`, `web_inspector.py`), and the
learned model is bolted onto each separately via a `--checkpoint` flag.
Adding a model means editing all three.

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

`gpu_arena.py` and `two_team_gpu_arena.py` are hardcoded to
`LearnedSpymaster`: they import it directly, call
`spymaster.model.predict_proba`, read `.own_reward`/`.miss_penalty`, and
call `expected_reward_and_best_n` themselves. Any new model that scores
the whole vocabulary -- which the planned baseline does -- gets either
the slow per-process path or a forked copy of a subtle lockstep loop.

`two_team_gpu_arena.py:124-146` does five things inline. Once features
vary per model, only the first and last belong to the arena:

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

`to_device` replaces the arena reaching in to set `spymaster.model.to(device)`
and `spymaster.device`. The single-board path routes through `score_batch`
with a one-element list, so there is exactly one scoring implementation
per model.

## Step 4: Cache rollouts, not features

`generate_training_data.py:383` computes
`feature_builder(board, clue, sims, turn_index)` and
`simulate_natural_stop(board, clue, guesser, sims)`, then saves only the
*featurized* row plus the outcome -- discarding the board, clue, turn
index, and which guesser was drawn.

This welds the expensive, model-independent half (simulating the guesser)
to the cheap, model-specific half (computing features). Generation is
"the dominant cost (~40 min sequentially at moderate scale)" per
`run_ablation_study.py`; training is fast. Since features now change
between models, every new model would re-simulate identical rollouts just
to get different columns out of them. `ablation.py` already exists as a
partial workaround -- it derives what it can by rearranging columns --
and the `unsorted`/`pool_*` ablations already need fresh generation
passes for the same reason.

Store the rollout instead: `(board, clue, guesser, turn_index) ->
(k, cause)`. Features are then computed on demand per model. A new
feature design costs a recompute, not a re-simulation, and every model
trains on the identical rollout set, which makes model-to-model
comparison cleaner.

**Built, and measured.** `codenames/rollouts.py` stores the rollouts;
`scripts/featurize_rollouts.py` turns a rollout set into a training
dataset whose on-disk layout is byte-compatible with the pre-split one,
so `scripts/train_scorer.py` needed no changes at all. On a 500-example
sample: generation runs at ~318 examples/sec, featurization at ~11,161 —
**~35x**, which is what a feature-design change now costs relative to a
regeneration. Storage is 4.1x smaller (54,134 bytes of rollouts vs
222,512 of features; earlier estimates of ~6x and ~60 bytes/row were
optimistic). Equivalence was verified rather than assumed: the
featurized rollouts reproduce the pre-refactor dataset exactly — all four
arrays (`features`, `outcome`, `reward`, `seed`) identical at the same
seed.

Reward is not stored, only derived, for the reason given above.

This is the same discipline as the similarity tensor and the LLM
response cache, applied to the third expensive model-independent thing.

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

**Identify a model by checkpoint content hash, not path.**
`train_scorer.py:259` writes `cache/checkpoints/scorer_best.pt` every
run, so a path-keyed cache would silently serve one model's expensive
results as another's.

Because the whole game replays through cached responses, and because the
cache key does not mention the spymaster, two models that give the same
clue on the same remaining words get the *identical* LLM response. That
makes cross-model comparison paired rather than independently sampled --
a methodological benefit, not just a cost saving.

## Step 7: Naming and layout

- Models get descriptive names, not version numbers. `v1`/`v1.1` keep
  their names for now and will be renamed or removed later; `CLAUDE.md`'s
  version-doc convention needs updating to match.
- `scripts/` groups into one-time data acquisition, the iteration loop,
  and interactive tools.
- Artifact directories get a systematic naming convention instead of
  `cache/m9/`, `cache/arena_blend.db`, `cache/sanity_check2.db`.

## Operational note

`cache/llm_store.db` is the record of every dollar spent, and since LLMs
are not deterministic even at temperature 0, it is also the only thing
making past evals reproducible. It lives in a gitignored directory with
no backup. It needs a real backup story.
