#!/usr/bin/env bash
# Four-way reward/observation ablation for the lib3 history-SAC question.
#
# Rows: H (pressure+action history) vs HR (the same history plus explicit rate)
# Cols: base tracking+identical safety bound vs phase-aware rate reward.
#
# The script never overwrites a model directory.  Use one seed per terminal:
#   bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-seed 0
#   bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-seed 1
#   bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-seed 2

set -euo pipefail

MODE="${1:-status}"
SEED="${2:-}"

ROOT=/home/rise/test/brl_window_v2
PYTHON=/home/rise/miniconda3/envs/env_1/bin/python
PROFILE="$ROOT/pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml"
# `_r025` deliberately distinguishes this clean experiment from the aborted
# 0.05 pre-calibration pilot, which is retained and must not be analysed.
PREFIX=260831_lib3_histreward2x2_r025
EPISODES="${EPISODES:-2000}"
OUTPUT_ROOT="pneu_rl/exp/lib3_0831_histreward2x2_fullbank"

# Locked screening objective.  The rate term is normalized by 12/6 kPa/s in
# the environment.  No action penalty is enabled here: adding it would create
# a third reward intervention.
RATE_COEFF=0.025
ACTION_DELTA_COEFF=0.0
SECOND_ACTION_DELTA_COEFF=0.0
# Soft operating-envelope term, identical in all four arms.  It prevents a
# rate-scale/return comparison from exploiting unlimited chamber pressure.
CHAMBER_BOUND_COEFF=0.10

cd "$ROOT"
mkdir -p /tmp/mpl-cache

name_for() {
    local arm="$1"
    local seed="$2"
    printf '%s_%s_seed%s' "$PREFIX" "$arm" "$seed"
}

common_args() {
    printf '%s\n' \
        --lib 3 \
        --variant history_action_sac \
        --ref-seed 61098 \
        --reference-profile "$PROFILE" \
        --episodes "$EPISODES" \
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
        --chamber-bound-rwd-coeff "$CHAMBER_BOUND_COEFF" \
        --chamber-reserve-rwd-coeff 0.0 \
        --conflict-rwd-coeff 0.0 \
        --future-rwd-coeff 0.0 \
        --action-delta-rwd-coeff "$ACTION_DELTA_COEFF" \
        --second-action-delta-rwd-coeff "$SECOND_ACTION_DELTA_COEFF" \
        --checkpoint-interval 100
}

run_arm() {
    local arm="$1"
    local seed="$2"
    local dry_run="$3"
    local name
    name="$(name_for "$arm" "$seed")"
    if [[ "$dry_run" != "1" && -e "pneu_rl/models/$name" ]]; then
        echo "refusing to overwrite existing model: pneu_rl/models/$name" >&2
        exit 3
    fi

    local -a args=()
    while IFS= read -r value; do args+=("$value"); done < <(common_args)
    args+=(--seed "$seed" --name "$name")

    case "$arm" in
        H_base)
            args+=(--rate-rwd-coeff 0.0)
            ;;
        H_smooth)
            args+=(--rate-rwd-coeff "$RATE_COEFF")
            ;;
        HR_base)
            args+=(--rate-rwd-coeff 0.0 --include-physics-features --physics-feature-mode rate)
            ;;
        HR_smooth)
            args+=(--rate-rwd-coeff "$RATE_COEFF" --include-physics-features --physics-feature-mode rate)
            ;;
        *)
            echo "unknown arm: $arm" >&2
            exit 2
            ;;
    esac
    if [[ "$dry_run" == "1" ]]; then
        args+=(--dry-run)
    fi

    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_screening.py "${args[@]}"
}

check_seed() {
    local seed="$1"
    for arm in H_base H_smooth HR_base HR_smooth; do
        local name
        name="$(name_for "$arm" "$seed")"
        if [[ -f "pneu_rl/models/$name/model.pth" ]]; then
            echo "READY   $name"
        else
            echo "MISSING $name"
        fi
    done
}

train_seed() {
    local seed="$1"
    for arm in H_base H_smooth HR_base HR_smooth; do
        run_arm "$arm" "$seed" 0
    done
}

train_arm() {
    local arm="$1"
    local seed="$2"
    run_arm "$arm" "$seed" 0
}

evaluate_seed() {
    local seed="$1"
    local -a args=()
    for arm in H_base H_smooth HR_base HR_smooth; do
        local name
        name="$(name_for "$arm" "$seed")"
        if [[ ! -f "pneu_rl/models/$name/model.pth" ]]; then
            echo "missing trained model: $name" >&2
            exit 4
        fi
        args+=(--model "$name")
    done
    env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl-cache \
        "$PYTHON" pneu_rl/scripts/evaluate_lib3_screening.py \
        "${args[@]}" \
        --steps 2048 \
        --all-reference-members \
        --no-plot \
        --output-root "$OUTPUT_ROOT/seed$seed"
}

case "$MODE" in
    status)
        for seed in 0 1 2; do check_seed "$seed"; done
        ;;
    dry-run)
        for arm in H_base H_smooth HR_base HR_smooth; do run_arm "$arm" 0 1; done
        ;;
    train-seed)
        [[ "$SEED" =~ ^[012]$ ]] || { echo "usage: $0 train-seed {0|1|2}" >&2; exit 2; }
        train_seed "$SEED"
        ;;
    train-arm)
        [[ "${2:-}" =~ ^(H_base|H_smooth|HR_base|HR_smooth)$ ]] || {
            echo "usage: $0 train-arm {H_base|H_smooth|HR_base|HR_smooth} {0|1|2}" >&2
            exit 2
        }
        [[ "${3:-}" =~ ^[012]$ ]] || {
            echo "usage: $0 train-arm {H_base|H_smooth|HR_base|HR_smooth} {0|1|2}" >&2
            exit 2
        }
        train_arm "$2" "$3"
        ;;
    train-all)
        for seed in 0 1 2; do train_seed "$seed"; done
        ;;
    evaluate-seed)
        [[ "$SEED" =~ ^[012]$ ]] || { echo "usage: $0 evaluate-seed {0|1|2}" >&2; exit 2; }
        evaluate_seed "$SEED"
        ;;
    *)
        echo "usage: $0 {status|dry-run|train-seed {0|1|2}|train-arm ARM SEED|train-all|evaluate-seed {0|1|2}}" >&2
        exit 2
        ;;
esac
