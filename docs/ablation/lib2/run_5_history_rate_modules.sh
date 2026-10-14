#!/usr/bin/env bash
set -euo pipefail

# Five new lib2 experiments, executed sequentially in this one terminal.
# Existing Pure SAC and old MPOBS+CAPS models are historical baselines and are
# intentionally not retrained here.

ROOT=/home/rise/test/brl_window_v2
PYTHON=/home/rise/miniconda3/envs/env_1/bin/python

cd "$ROOT"
mkdir -p /tmp/mpl-cache

run_train() {
    local name="$1"
    shift
    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_screening.py \
        --lib 2 \
        --seed 0 \
        --ref-seed 61098 \
        --reference-kind legacy_random \
        --episodes 1000 \
        --horizon 512 \
        --carry \
        --start-epi 10 \
        --gamma 0.9 \
        --scale-low 0.8 \
        --action-history-len 10 \
        --exploration none \
        --buffer-size 500000 \
        --normalize-observation \
        --warmup-mode fixed \
        --warmup-physical-action 0.95 \
        --defer-updates-during-warmup \
        --reset-after-warmup \
        --tracking-loss abs \
        --future-rwd-coeff 0.0 \
        --name "$name" \
        "$@"
}

# 3. History + action history + pressure-rate observation.
run_train \
    260811_lib2_history_action_rate_seed0 \
    --variant history_action_rate \
    --include-physics-features \
    --physics-feature-mode rate

# 4. Previous model + K=2 action chunking.
run_train \
    260811_lib2_history_action_rate_k2_seed0 \
    --variant history_action_rate_k2 \
    --include-physics-features \
    --physics-feature-mode rate \
    --action-chunk-steps 2

# 5. K=2 + small mean-action CAPS.
run_train \
    260811_lib2_history_action_rate_k2_caps001_seed0 \
    --variant history_action_rate_k2_caps \
    --include-physics-features \
    --physics-feature-mode rate \
    --action-chunk-steps 2 \
    --caps-mode mean_action \
    --caps-temporal-weight 0.01 \
    --caps-spatial-weight 0.01 \
    --caps-noise-std 0.0 \
    --caps-start-after-warmup

# 6. K=2 + MPOBS1.
run_train \
    260811_lib2_history_action_rate_k2_mpobs1_seed0 \
    --variant history_action_rate_k2_mpobs1 \
    --include-physics-features \
    --physics-feature-mode rate \
    --action-chunk-steps 2

# 7. K=2 + MPOBS1 + small mean-action CAPS.
run_train \
    260811_lib2_history_action_rate_k2_mpobs1_caps001_seed0 \
    --variant history_action_rate_k2_mpobs1_caps \
    --include-physics-features \
    --physics-feature-mode rate \
    --action-chunk-steps 2 \
    --caps-mode mean_action \
    --caps-temporal-weight 0.01 \
    --caps-spatial-weight 0.01 \
    --caps-noise-std 0.0 \
    --caps-start-after-warmup
