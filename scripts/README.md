# scripts/

Grouped by *when you run them*, which is the distinction that actually
matters here — a one-time embedding download and a per-iteration training run
had been sitting side by side in one flat directory of 22 files.

Scripts import their siblings via `sys.path` (`Path(__file__).parent`), so
a script that imports another lives in the same group as the one it
imports. That constraint is why the grouping is what it is.

## `data/` — build time, run once

Acquire and prepare the inputs everything else reads. After
`build_similarity_tensor.py`, the embedding models are never loaded again.

| script | what it does |
|---|---|
| `download_embeddings.py` | fetch the pretrained embedding spaces |
| `build_similarity_tensor.py` | the `[n_clues × n_board_words × n_spaces]` tensor |
| `extend_similarity_tensor.py` | add one space to an existing tensor |
| `sanity_check_sims.py` | eyeball the built tensor |
| `_embedding_lib.py` | shared loading helpers (not a CLI) |

## `pipeline/` — the iteration loop

What you run when trying a new model. See
[`docs/iteration-architecture.md`](../docs/iteration-architecture.md).

| script | what it does |
|---|---|
| `generate_training_data.py` | simulate guesser rollouts → `cache/rollouts` |
| `featurize_rollouts.py` | rollouts → a model's feature dataset (~35x cheaper than regenerating) |
| `train_scorer.py` | train on a feature dataset → a checkpoint |
| `run_arena.py` | cross-play matrix vs. synthetic guessers — a **training diagnostic, not a scoreboard** |
| `run_two_team_arena.py` | two-team self-play |
| `run_ablation_study.py` | generate + train a batch of variants, with a comparison report |

Evaluation proper — the LLM guesser, against frozen board seeds — lives in
`codenames/eval_suite.py`, not here.

## `tools/` — interactive, run whenever

| script | what it does |
|---|---|
| `web_inspector.py` | web UI: play a full two-team game, or inspect one clue with live controls |
| `inspector.py` | CLI single-turn inspector |
| `check_clue.py` | look up one clue's similarities |
| `dump_game_records.py` | human-readable view of recorded games |
| `probe_llm_cost.py` | price an LLM guesser turn on real positions (**spends money**; `--dry-run` is free) |
