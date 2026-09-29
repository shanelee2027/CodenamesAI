#!/usr/bin/env bash
# The whole win_actor_critic pipeline, in order (docs/versions/win_actor_critic.md).
#
#   setsid nohup scripts/pipeline/run_win_actor_critic.sh > cache/training_data/win_pipeline.log 2>&1 &
#
# Each step leaves a marker in cache/training_data/win_steps/ when it ends,
# including when it ends because its budget ran out, and a rerun skips marked
# steps. So resuming never renews a budget. The actor-critic step carries its
# spend in its own checkpoint.
#
# Guesser budgets (gpt-oss-120b on DeepInfra, metered per step by
# clue_policy.GuesserMeter) add up to $7.1 on a $10 account. The rest is
# margin for the meter's per-token price estimate. The last step plays the
# frozen eval suite with Sonnet, on the Anthropic key.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
T=scripts/pipeline/train_win_actor_critic.py
D=cache/training_data
M=$D/win_steps
mkdir -p $M
backup() { cp cache/llm_store.db "cache/backups/llm_store_$(date +%F)_pre_win_$1.db"; }
step() { # name, command...: run once, then mark it done
    local name=$1; shift
    if [ -f "$M/$name" ]; then echo "== $name: already done"; return; fi
    echo "== $(date) $name"
    "$@"
    date > "$M/$name"
}

echo "== $(date) waiting for the uncapped imitation data"
while [ ! -f $D/policy_imitation_train_uncapped.npz ]; do sleep 60; done

# 1. Imitation warm start: the incumbent's (clue, k) picks only. Free.
step 1_imitation $PY scripts/pipeline/train_imitation_policy.py --head k --epochs 10 \
    --out cache/win_policy_init.pt

# 2. Critic pilot: the initial policy, sampling, on training seeds. <= $1.3.
backup pilot
step 2_pilot_games $PY $T play --policy cache/win_policy_init.pt --seeds train --n 2000 --budget 1.3 \
    --out $D/win_games_pilot.jsonl

# 3. The initial policy, greedy, on validation seeds: the "before". <= $0.3.
step 3_val_init $PY $T play --policy cache/win_policy_init.pt --seeds val --n 300 --greedy --budget 0.3 \
    --out $D/win_games_val_init.jsonl

# 4. Critic fit on the pilot games. Free.
step 4_critic $PY $T fit-critic $D/win_games_pilot.jsonl --final

# 5. Actor-critic. <= $5.2 in total, across resumes.
backup train
step 5_train $PY $T train --budget 5.2 --resume

# 6. The trained policy, greedy, on the same validation seeds: the "after". <= $0.3.
step 6_val_trained $PY $T play --policy cache/win_policy.pt --seeds val --n 300 --greedy --budget 0.3 \
    --out $D/win_games_val_trained.jsonl

# 7. Frozen eval suite, Sonnet guessing, against the incumbent.
backup sonnet
step 7_sonnet $PY scripts/pipeline/run_eval_suite.py win_actor_critic learned_listener

echo "== $(date) done"
