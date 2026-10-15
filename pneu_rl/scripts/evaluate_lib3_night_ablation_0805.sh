#!/usr/bin/env bash
# Deterministic one-shot validation for the six overnight lib3 models.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/rise/miniconda3/envs/env_1/bin/python}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pneu_rl/exp/260805_lib3_night_validation_2048}"

MODELS=(
    "260805_lib3_512_history_sac_protocol2_seed${SEED}"
    "260805_lib3_512_history_caps2_protocol2_seed${SEED}"
    "260805_lib3_512_mpobs1_sac_protocol2_seed${SEED}"
    "260805_lib3_512_mpobs1_caps2_protocol2_seed${SEED}"
    "260805_lib3_512_history_action_sac_protocol2_seed${SEED}"
    "260805_lib3_512_history_action_caps2_protocol2_seed${SEED}"
)

ARGS=()
for model in "${MODELS[@]}"; do
    if [[ ! -f "pneu_rl/models/${model}/cfg.yaml" ]]; then
        echo "missing completed model config: pneu_rl/models/${model}/cfg.yaml" >&2
        exit 3
    fi
    ARGS+=(--model "${model}")
done

OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 \
"${PYTHON_BIN}" pneu_rl/scripts/evaluate_lib3_screening.py \
    "${ARGS[@]}" \
    --episodes 1 \
    --steps 2048 \
    --tolerance 5.0 \
    --output-root "${OUTPUT_ROOT}"
