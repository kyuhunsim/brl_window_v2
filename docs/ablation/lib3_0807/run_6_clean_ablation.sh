#!/usr/bin/env bash

# lib3 clean two-reference ablation: 3 methods x 2 references.
# This file only provides commands and is intentionally not executed by Codex.

set -e

PYTHON=/home/rise/miniconda3/envs/env_1/bin/python
ROOT=/home/rise/test/brl_window_v2
PROFILE=pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml

cd "$ROOT"
mkdir -p /tmp/mpl-cache

COMMON=(
  --lib 3
  --seed 0
  --ref-seed 61098
  --episodes 1000
  --horizon 512
  --carry
  --carry-reset-blocks 4
  --start-epi 10
  --gamma 0.99
  --scale-low 0.8
  --action-history-len 10
  --exploration none
  --buffer-size 500000
)

run_train() {
  env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl-cache OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "$PYTHON" pneu_rl/scripts/train_screening.py "$@"
}

# 1. Latest feasible profile: clean history SAC baseline
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --reference-profile "$PROFILE" \
  --name 260808_lib3_clean_profile_history_sac_seed0

# 2. Latest feasible profile: add CAPS only
run_train "${COMMON[@]}" \
  --variant history_action_sac+caps \
  --reference-profile "$PROFILE" \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_clean_profile_history_caps_seed0

# 3. Latest feasible profile: add MPOBS1 on top of history SAC + CAPS
run_train "${COMMON[@]}" \
  --variant history_action_sac+mpobs1+caps \
  --reference-profile "$PROFILE" \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_clean_profile_history_mpobs1_caps_seed0

# 4. Legacy RandomRef: clean history SAC baseline
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --reference-kind legacy_random \
  --name 260808_lib3_clean_legacy_history_sac_seed0

# 5. Legacy RandomRef: add CAPS only
run_train "${COMMON[@]}" \
  --variant history_action_sac+caps \
  --reference-kind legacy_random \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_clean_legacy_history_caps_seed0

# 6. Legacy RandomRef: add MPOBS1 on top of history SAC + CAPS
run_train "${COMMON[@]}" \
  --variant history_action_sac+mpobs1+caps \
  --reference-kind legacy_random \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_clean_legacy_history_mpobs1_caps_seed0
