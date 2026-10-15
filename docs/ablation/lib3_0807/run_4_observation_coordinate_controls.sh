#!/usr/bin/env bash
set -euo pipefail

# Lib3 observation-coordinate controls: the core four-way comparison plus a
# paired five-step short-history compression check.

MODE="${1:-status}"
ROOT=/home/rise/test/brl_window_v2
PYTHON=/home/rise/miniconda3/envs/env_1/bin/python
PROFILE="$ROOT/pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml"

HISTORY=260810_lib3_ablation_profile_hist_base_seed0
RATE_CURRENT=260810_lib3_ablation_profile_hist_rate_seed0
RATE_MATCHED=260825_lib3_obscoord_hist_rate_scale5000_seed0
ZERO4=260825_lib3_obscoord_hist_zero4_seed0
SHORT5=260825_lib3_obscoord_short_hist5_base_seed0
SHORT5_RATE=260825_lib3_obscoord_short_hist5_rate_scale5000_seed0
OUTPUT_ROOT=pneu_rl/exp/lib3_0825_obscoord_6model_validation_actual2048
FULLBANK_OUTPUT_ROOT=pneu_rl/exp/lib3_0825_obscoord_6model_fullbank_actual2048

cd "$ROOT"
mkdir -p /tmp/mpl-cache

COMMON=(
    --lib 3
    --variant history_action_sac
    --seed 0
    --ref-seed 61098
    --reference-profile "$PROFILE"
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
    --normalize-observation
    --warmup-mode fixed
    --warmup-physical-action 0.95
    --defer-updates-during-warmup
    --reset-after-warmup
    --tracking-loss abs
    --rate-rwd-coeff 0.0
    --action-delta-rwd-coeff 0.0
    --second-action-delta-rwd-coeff 0.0
    --conflict-rwd-coeff 0.0
    --chamber-reserve-rwd-coeff 0.0
    --chamber-bound-rwd-coeff 0.0
    --future-rwd-coeff 0.0
    --checkpoint-interval 100
)

run_train() {
    local name="$1"
    shift
    if [[ -e "pneu_rl/models/$name" ]]; then
        echo "refusing to overwrite existing model: pneu_rl/models/$name" >&2
        exit 3
    fi
    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_screening.py \
        "${COMMON[@]}" \
        --name "$name" \
        "$@"
}

dry_run() {
    local name="$1"
    shift
    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_screening.py \
        "${COMMON[@]}" \
        --name "$name" \
        --dry-run \
        "$@"
}

print_status() {
    for name in \
        "$HISTORY" \
        "$RATE_CURRENT" \
        "$RATE_MATCHED" \
        "$ZERO4" \
        "$SHORT5" \
        "$SHORT5_RATE"; do
        if [[ -f "pneu_rl/models/$name/model.pth" ]]; then
            echo "READY   $name"
        else
            echo "MISSING $name"
        fi
    done
}

case "$MODE" in
    status)
        print_status
        ;;
    dry-run)
        dry_run "$RATE_MATCHED" \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        dry_run "$ZERO4" \
            --include-physics-features \
            --physics-feature-mode zero4 \
            --physics-rate-scale-kpa-s 50
        dry_run "$SHORT5" \
            --num-prev 5 \
            --action-history-len 5
        dry_run "$SHORT5_RATE" \
            --num-prev 5 \
            --action-history-len 5 \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        ;;
    train-scale-matched)
        run_train "$RATE_MATCHED" \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        ;;
    train-zero4)
        run_train "$ZERO4" \
            --include-physics-features \
            --physics-feature-mode zero4 \
            --physics-rate-scale-kpa-s 50
        ;;
    train-new)
        run_train "$RATE_MATCHED" \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        run_train "$ZERO4" \
            --include-physics-features \
            --physics-feature-mode zero4 \
            --physics-rate-scale-kpa-s 50
        ;;
    train-short-history)
        run_train "$SHORT5" \
            --num-prev 5 \
            --action-history-len 5
        ;;
    train-short-history-rate)
        run_train "$SHORT5_RATE" \
            --num-prev 5 \
            --action-history-len 5 \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        ;;
    train-short-pair)
        run_train "$SHORT5" \
            --num-prev 5 \
            --action-history-len 5
        run_train "$SHORT5_RATE" \
            --num-prev 5 \
            --action-history-len 5 \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        ;;
    train-all-new)
        run_train "$RATE_MATCHED" \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        run_train "$ZERO4" \
            --include-physics-features \
            --physics-feature-mode zero4 \
            --physics-rate-scale-kpa-s 50
        run_train "$SHORT5" \
            --num-prev 5 \
            --action-history-len 5
        run_train "$SHORT5_RATE" \
            --num-prev 5 \
            --action-history-len 5 \
            --include-physics-features \
            --physics-feature-mode rate \
            --physics-rate-scale-kpa-s 5000
        ;;
    validate)
        print_status
        for name in \
            "$HISTORY" \
            "$RATE_CURRENT" \
            "$RATE_MATCHED" \
            "$ZERO4" \
            "$SHORT5" \
            "$SHORT5_RATE"; do
            if [[ ! -f "pneu_rl/models/$name/model.pth" ]]; then
                echo "cannot validate until all four model.pth files exist" >&2
                exit 4
            fi
        done
        env -u PYTHONPATH \
            MPLCONFIGDIR=/tmp/mpl-cache \
            "$PYTHON" pneu_rl/scripts/evaluate_lib3_screening.py \
            --model "$HISTORY" \
            --model "$RATE_CURRENT" \
            --model "$RATE_MATCHED" \
            --model "$ZERO4" \
            --model "$SHORT5" \
            --model "$SHORT5_RATE" \
            --episodes 1 \
            --steps 2048 \
            --output-root "$OUTPUT_ROOT"
        ;;
    validate-fullbank)
        print_status
        for name in \
            "$HISTORY" \
            "$RATE_CURRENT" \
            "$RATE_MATCHED" \
            "$ZERO4" \
            "$SHORT5" \
            "$SHORT5_RATE"; do
            if [[ ! -f "pneu_rl/models/$name/model.pth" ]]; then
                echo "cannot validate until all six model.pth files exist" >&2
                exit 4
            fi
        done
        env -u PYTHONPATH \
            MPLCONFIGDIR=/tmp/mpl-cache \
            "$PYTHON" pneu_rl/scripts/evaluate_lib3_screening.py \
            --model "$HISTORY" \
            --model "$RATE_CURRENT" \
            --model "$RATE_MATCHED" \
            --model "$ZERO4" \
            --model "$SHORT5" \
            --model "$SHORT5_RATE" \
            --steps 2048 \
            --all-reference-members \
            --no-plot \
            --output-root "$FULLBANK_OUTPUT_ROOT"
        ;;
    *)
        echo "usage: bash $0 {status|dry-run|train-scale-matched|train-zero4|train-new|train-short-history|train-short-history-rate|train-short-pair|train-all-new|validate|validate-fullbank}" >&2
        exit 2
        ;;
esac
