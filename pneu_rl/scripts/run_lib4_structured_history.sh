#!/usr/bin/env bash
set -euo pipefail

# Matched lib4 structured-history setup:
#   HRR   = history + pressure rate + chamber/actuator reserve
#   PIDE  = tracking error + integral error + error derivative
#   GPIDE = generalized observation difference + action + action difference
#
# The defaults match the existing lib4 screening protocol.  Set DRY_RUN=1 to
# validate all observation dimensions without starting a training run.

ROOT="${ROOT:-/home/rise/test/brl_window_v2}"
PYTHON="${PYTHON:-/home/rise/miniconda3/envs/env_1/bin/python}"
PROFILE="${PROFILE:-$ROOT/pneu_rl/config/reference/lib4_capacity_core_16x4x512_seed64001.yaml}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-1000}"
DRY_RUN="${DRY_RUN:-0}"

cd "$ROOT"
mkdir -p /tmp/mpl-cache-lib4-structured

run_one() {
    local variant="$1"
    local name="260902_lib4_${variant}_seed${SEED}"
    local dry=()
    if [[ "$DRY_RUN" == "1" ]]; then
        dry=(--dry-run)
    fi
    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache-lib4-structured \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_lib4_screening.py \
        --variant "$variant" \
        --reference-profile "$PROFILE" \
        --seed "$SEED" \
        --episodes "$EPISODES" \
        --physical-horizon 512 \
        --carry-blocks 4 \
        --history-len 10 \
        --reference-lookahead 15 \
        --start-epi 10 \
        --pos-reward 0.3 \
        --neg-reward 0.3 \
        --disp-reward 0.2 \
        --diff-reward 0.0 \
        --learning-rate 3e-4 \
        --gamma 0.99 \
        --tau 0.005 \
        --alpha 0.5 \
        --hidden-dim 256 \
        --buffer-size 500000 \
        --batch-size 128 \
        --name "$name" \
        "${dry[@]}"
}

for variant in hrr pide gpide; do
    run_one "$variant"
done

