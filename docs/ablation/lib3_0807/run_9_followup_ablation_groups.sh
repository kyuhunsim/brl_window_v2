#!/usr/bin/env bash

# Follow-up 9-run ablation for groups 3, 4, and 5.
# Run one group per terminal:
#   bash run_9_followup_ablation_groups.sh 1
#   bash run_9_followup_ablation_groups.sh 2
#   bash run_9_followup_ablation_groups.sh 3
# Each group runs three experiments sequentially.
# This file only provides commands and is intentionally not executed by Codex.

set -e

if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[1-3]$ ]]; then
  echo "usage: $0 {1|2|3}" >&2
  exit 2
fi

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

case "$1" in
  1)
    # 7. K=1 + physics observation + integral error.
    # The separate K=1 physics and K=1 integral runs already exist.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --include-physics-features \
      --include-integral-error \
      --integral-error-decay 1.0 \
      --integral-error-limit 100.0 \
      --integral-error-scale 20.0 \
      --reference-profile "$PROFILE" \
      --name 260808_lib3_ablation_profile_k1_physics_integral_seed0

    # 8. K=2 baseline with the latest feasible profile.
    # The previous K=2 baseline used rate_limited_random.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --action-chunk-steps 2 \
      --reference-profile "$PROFILE" \
      --name 260808_lib3_ablation_profile_k2_baseline_seed0

    # 9. K=2 + physics with the latest feasible profile.
    # The previous K=2 physics run used rate_limited_random.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --action-chunk-steps 2 \
      --include-physics-features \
      --reference-profile "$PROFILE" \
      --name 260808_lib3_ablation_profile_k2_physics_seed0
    ;;

  2)
    # 10. Legacy K=1 + physics observation.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --include-physics-features \
      --reference-kind legacy_random \
      --name 260808_lib3_ablation_legacy_k1_physics_seed0

    # 11. Legacy K=1 + integral error.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --include-integral-error \
      --integral-error-decay 1.0 \
      --integral-error-limit 100.0 \
      --integral-error-scale 20.0 \
      --reference-kind legacy_random \
      --name 260808_lib3_ablation_legacy_k1_integral_seed0

    # 12. Legacy K=1 + physics observation + integral error.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --include-physics-features \
      --include-integral-error \
      --integral-error-decay 1.0 \
      --integral-error-limit 100.0 \
      --integral-error-scale 20.0 \
      --reference-kind legacy_random \
      --name 260808_lib3_ablation_legacy_k1_physics_integral_seed0
    ;;

  3)
    # 13. Legacy K=2 + physics observation.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --action-chunk-steps 2 \
      --include-physics-features \
      --reference-kind legacy_random \
      --name 260808_lib3_ablation_legacy_k2_physics_seed0

    # 14. Legacy K=2 + physics observation + integral error.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --action-chunk-steps 2 \
      --include-physics-features \
      --include-integral-error \
      --integral-error-decay 1.0 \
      --integral-error-limit 100.0 \
      --integral-error-scale 20.0 \
      --reference-kind legacy_random \
      --name 260808_lib3_ablation_legacy_k2_physics_integral_seed0

    # 15. Latest feasible profile K=2 + physics + integral, train seed 1.
    # Seed 0 already exists as 260807_lib3_k2_integral_profile_physics_seed0.
    run_train "${COMMON[@]}" \
      --variant history_action_sac \
      --action-chunk-steps 2 \
      --include-physics-features \
      --include-integral-error \
      --integral-error-decay 1.0 \
      --integral-error-limit 100.0 \
      --integral-error-scale 20.0 \
      --reference-profile "$PROFILE" \
      --seed 1 \
      --name 260808_lib3_ablation_profile_k2_physics_integral_seed1
    ;;
esac
