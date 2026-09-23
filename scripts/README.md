# scripts/

Grouped by *when you run them*. Every script imports the `codenames` package
through the editable install (`pip install -e .`), so run them with
`.venv/bin/python` from anywhere. Reusable logic lives in the package, not in
a script: nothing here imports another script.

Scripts that spend money say so in their docstring and have a `--dry-run`.

## `data/`: build time, and buying teacher data

| script | what it does |
|---|---|
| `download_embeddings.py` | fetch the pretrained embedding files |
| `build_similarity_tensor.py` | the `[clues × board words × spaces]` tensor (GloVe slice) |
| `extend_similarity_tensor.py` | add Numberbatch and Wikipedia2Vec to the tensor |
| `_embedding_lib.py` | shared embedding-loading helpers (not a CLI) |
| `sanity_check_sims.py` | print the nearest clues for sample board words |
| `build_clue_stats.py` | per-clue mean, spread and rarity, for z-scores |
| `build_acronym_mask.py` | flag acronyms so spymasters can refuse to play them |
| `build_swow_tables.py` | human word-association strengths, both directions |
| `build_lm_pmi.py` | language-model PMI between every clue and board word (GPU) |
| `build_extra_space_sims.py` | GloVe-840B and fastText similarity side tables |
| `build_entity_sims.py` | Wikipedia2Vec entity-vector similarities |
| `build_word_norms.py` | concreteness, familiarity, frequency per board word |
| `build_wordnet_sims.py` | WordNet Wu–Palmer similarity |
| `build_lexical_sims.py` | surface-form and gloss links between clue and word |
| `collect_listener_data.py` | buy LLM rankings to train the listener (**spends money**) |
| `collect_decoy_data.py` | buy rankings with off-board decoys mixed in (**spends money**) |

## `pipeline/`: train and evaluate

| script | what it does |
|---|---|
| `train_listener.py` | distil the listener from cached rankings |
| `run_two_team_arena.py` | two-team games: self-play, or head-to-head with `--vs` |
| `run_eval_suite.py` | the frozen benchmark, head to head against the suite's guesser (**spends money** on a new pair) |

## `tools/`: analysis, sweeps, play

| script | what it does |
|---|---|
| `play_server.py` | play against any registered spymaster on localhost; `/eval` is the blind human study |
| `analyze_human_eval.py` | compare the study's arms |
| `make_demo_bundle.py` | pack a self-contained laptop demo |
| `sweep_role_costs.py` | sweep the incumbent's role costs / outside option head to head |
| `analyze_headtohead.py` | board-paired statistics for recorded matchups |
| `dump_game_records.py` | print a recorded game's transcript |
| `compare_guesser_models.py` | compare guesser models on the same real turns |
| `sweep_sigma.py` | per-turn σ sweep for `expected_words` against a real guesser |
| `fit_listener_sigma.py` | the σ that best describes cached rankings |
| `clue_number_arc.py` | how the announced number falls over a game, per σ (plotted in `scratch/`) |
| `eval_listener_calibration.py` | score the listener as a probability model |
| `eval_listener_holdout.py` | score the listener once on untouched boards |
| `sweep_listener_params.py` | listener hyperparameter sweep |
| `prune_listener_features.py` | leave-one-out feature pruning |
| `probe_anchors.py` | does PASS or a decoy carry signal about clue quality? |
| `analyze_decoy_invariance.py` | is the decoy signal invariant to the decoy count? |
| `fit_decoy_level.py` | fit the decoy level and test it across decoy counts |
| `render_game_tikz.py` | a recorded game as TikZ, for the paper |
| `export_board_inputs.py`, `export_listener_js.py` | data for the browser port in `webport/` (the phone artifact) |
