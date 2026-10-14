#!/usr/bin/env bash

# lib3 10-run ablation setup.
# This script only launches training; it is intentionally not executed by Codex.
# Use one command at a time or run the script sequentially.

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

# 1. Feasible profile: history SAC
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --reference-profile "$PROFILE" \
  --name 260808_lib3_ablation_profile_history_sac_seed0

# 2. Feasible profile: history SAC + CAPS
run_train "${COMMON[@]}" \
  --variant history_action_sac+caps \
  --reference-profile "$PROFILE" \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_ablation_profile_history_caps_seed0

# 3. Feasible profile: history SAC + MPOBS1 + CAPS
run_train "${COMMON[@]}" \
  --variant history_action_sac+mpobs1+caps \
  --reference-profile "$PROFILE" \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_ablation_profile_history_mpobs1_caps_seed0

# 4. Legacy RandomRef: history SAC
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --reference-kind legacy_random \
  --name 260808_lib3_ablation_legacy_history_sac_seed0

# 5. Legacy RandomRef: history SAC + CAPS
run_train "${COMMON[@]}" \
  --variant history_action_sac+caps \
  --reference-kind legacy_random \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_ablation_legacy_history_caps_seed0

# 6. Legacy RandomRef: history SAC + MPOBS1 + CAPS
run_train "${COMMON[@]}" \
  --variant history_action_sac+mpobs1+caps \
  --reference-kind legacy_random \
  --caps-mode legacy \
  --caps-temporal-weight 0.05 \
  --caps-spatial-weight 0.05 \
  --caps-noise-std 0.0 \
  --name 260808_lib3_ablation_legacy_history_mpobs1_caps_seed0

# 7. Feasible profile: K=2 history SAC baseline
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --action-chunk-steps 2 \
  --reference-profile "$PROFILE" \
  --name 260808_lib3_ablation_profile_k2_baseline_seed0

# 8. Feasible profile: K=2 + physics observation
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --action-chunk-steps 2 \
  --include-physics-features \
  --reference-profile "$PROFILE" \
  --name 260808_lib3_ablation_profile_k2_physics_seed0

# 9. Feasible profile: K=2 + integral error
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --action-chunk-steps 2 \
  --include-integral-error \
  --integral-error-decay 1.0 \
  --integral-error-limit 100.0 \
  --integral-error-scale 20.0 \
  --reference-profile "$PROFILE" \
  --name 260808_lib3_ablation_profile_k2_integral_seed0

# 10. Feasible profile: K=2 + physics + integral + chamber bound
run_train "${COMMON[@]}" \
  --variant history_action_sac \
  --action-chunk-steps 2 \
  --include-physics-features \
  --include-integral-error \
  --integral-error-decay 1.0 \
  --integral-error-limit 100.0 \
  --integral-error-scale 20.0 \
  --chamber-bound-rwd-coeff 1.0 \
  --reference-profile "$PROFILE" \
  --name 260808_lib3_ablation_profile_k2_physics_integral_bound_seed0
