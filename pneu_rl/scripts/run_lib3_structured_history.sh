#!/usr/bin/env bash
set -euo pipefail

# Matched lib3 structured-history setup.  Runs use the existing history_action
# SAC policy; only the deterministic physics feature changes.
#
#   hrr   = pressure rate + chamber/actuator reserve
#   pide  = tracking error + integral error + error derivative
#   gpide = observation difference + current action + action difference
#
# GPIDE is optional here: for lib3's pressure-only observation, its observation
# difference is already pressure-rate-like and the action terms overlap with
# history_action_sac.  The first screening therefore defaults to HRR/PIDE/e_dot.
#
# By default all three modes run sequentially.  To split across two terminals:
#   MODES=hrr,pide bash pneu_rl/scripts/run_lib3_structured_history.sh
#   MODES=gpide bash pneu_rl/scripts/run_lib3_structured_history.sh
# Set DRY_RUN=1 before the first training run to validate dimensions/reward.

ROOT="${ROOT:-/home/rise/test/brl_window_v2}"
PYTHON="${PYTHON:-/home/rise/miniconda3/envs/env_1/bin/python}"
PROFILE="${PROFILE:-$ROOT/pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.yaml}"
SEED="${SEED:-0}"
REF_SEED="${REF_SEED:-61098}"
EPISODES="${EPISODES:-1000}"
DRY_RUN="${DRY_RUN:-0}"
MODES="${MODES:-hrr,pide,edot}"

cd "$ROOT"
mkdir -p /tmp/mpl-cache-lib3-structured

run_one() {
    local mode="$1"
    local name="260906_lib3_${mode}_seed${SEED}"
    local dry=()
    if [[ "$DRY_RUN" == "1" ]]; then
        dry=(--dry-run)
    fi
    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache-lib3-structured \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_screening.py \
        --lib 3 \
        --variant history_action_sac \
        --seed "$SEED" \
        --ref-seed "$REF_SEED" \
        --reference-profile "$PROFILE" \
        --episodes "$EPISODES" \
        --horizon 512 \
        --carry \
        --carry-reset-blocks 4 \
        --start-epi 10 \
        --gamma 0.99 \
        --scale-low 0.8 \
        --defer-updates-during-warmup \
        --reset-after-warmup \
        --chamber-bound-rwd-coeff 0.10 \
        --chamber-reserve-rwd-coeff 0.0 \
        --conflict-rwd-coeff 0.0 \
        --future-rwd-coeff 0.0 \
        --include-physics-features \
        --physics-feature-mode "$mode" \
        --physics-rate-scale-kpa-s 50.0 \
        --action-history-len 10 \
        --normalize-observation \
        --checkpoint-interval 100 \
        --name "$name" \
        "${dry[@]}"
}

IFS=',' read -r -a mode_list <<< "$MODES"
for mode in "${mode_list[@]}"; do
    case "$mode" in
        hrr|pide|gpide|edot|rate_reserve_edot) ;;
        *)
            echo "unsupported mode '$mode'; choose hrr,pide,gpide,edot,rate_reserve_edot" >&2
            exit 2
            ;;
    esac
    run_one "$mode"
done
