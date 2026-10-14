#!/usr/bin/env bash
set -euo pipefail

# Run one group in one terminal.  Each group contains exactly three
# experiments and is executed sequentially.  Use groups 1, 2, and 3 in three
# separate terminals to run the nine experiments in parallel by group.

GROUP="${1:-}"
if [[ "$GROUP" != "1" && "$GROUP" != "2" && "$GROUP" != "3" ]]; then
    echo "usage: bash $0 {1|2|3}" >&2
    exit 2
fi

ROOT=/home/rise/test/brl_window_v2
PYTHON=/home/rise/miniconda3/envs/env_1/bin/python
PROFILE="$ROOT/pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml"

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
        --lib 3 \
        --seed 0 \
        --ref-seed 61098 \
        --reference-profile "$PROFILE" \
        --episodes 1000 \
        --horizon 512 \
        --carry \
        --carry-reset-blocks 4 \
        --start-epi 10 \
        --gamma 0.99 \
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
        --rate-rwd-coeff 0.0 \
        --action-delta-rwd-coeff 0.0 \
        --second-action-delta-rwd-coeff 0.0 \
        --conflict-rwd-coeff 0.0 \
        --chamber-reserve-rwd-coeff 0.0 \
        --chamber-bound-rwd-coeff 0.0 \
        --future-rwd-coeff 0.0 \
        --name "$name" \
        "$@"
}

case "$GROUP" in
    1)
        # Base, pressure-rate-only, chamber-reserve-only.
        run_train \
            260810_lib3_ablation_profile_hist_base_seed0 \
            --variant history_action_sac
        run_train \
            260810_lib3_ablation_profile_hist_rate_seed0 \
            --variant history_action_sac \
            --include-physics-features \
            --physics-feature-mode rate
        run_train \
            260810_lib3_ablation_profile_hist_reserve_seed0 \
            --variant history_action_sac \
            --include-physics-features \
            --physics-feature-mode reserve
        ;;
    2)
        # Rate+reserve, then integral-error and K=2 action chunking.
        run_train \
            260810_lib3_ablation_profile_hist_rate_reserve_seed0 \
            --variant history_action_sac \
            --include-physics-features \
            --physics-feature-mode rate_reserve
        run_train \
            260810_lib3_ablation_profile_hist_rate_reserve_integral_seed0 \
            --variant history_action_sac \
            --include-physics-features \
            --physics-feature-mode rate_reserve \
            --include-integral-error \
            --integral-error-decay 1.0 \
            --integral-error-limit 100.0 \
            --integral-error-scale 20.0
        run_train \
            260810_lib3_ablation_profile_hist_rate_reserve_k2_seed0 \
            --variant history_action_sac \
            --include-physics-features \
            --physics-feature-mode rate_reserve \
            --action-chunk-steps 2
        ;;
    3)
        # CAPS and MPOBS effects after the complete rate+reserve observation.
        run_train \
            260810_lib3_ablation_profile_hist_rate_reserve_caps001_seed0 \
            --variant history_action_sac+caps \
            --include-physics-features \
            --physics-feature-mode rate_reserve \
            --caps-mode mean_action \
            --caps-temporal-weight 0.01 \
            --caps-spatial-weight 0.01 \
            --caps-noise-std 0.0
        run_train \
            260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_seed0 \
            --variant history_action_sac+mpobs1 \
            --include-physics-features \
            --physics-feature-mode rate_reserve
        run_train \
            260810_lib3_ablation_profile_hist_rate_reserve_mpobs1_caps001_seed0 \
            --variant history_action_sac+mpobs1+caps \
            --include-physics-features \
            --physics-feature-mode rate_reserve \
            --caps-mode mean_action \
            --caps-temporal-weight 0.01 \
            --caps-spatial-weight 0.01 \
            --caps-noise-std 0.0
        ;;
esac
