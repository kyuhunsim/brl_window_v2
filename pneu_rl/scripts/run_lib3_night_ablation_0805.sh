#!/usr/bin/env bash
# Run one member of the repaired 3-observation x 2-CAPS lib3 matrix.
# This script never launches the other five jobs and does not use screen/tail.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 {1|2|3|4|5|6}" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/rise/miniconda3/envs/env_1/bin/python}"
EPISODES="${EPISODES:-1000}"
SEED="${SEED:-0}"
REF_SEED="${REF_SEED:-61098}"

COMMON=(
    pneu_rl/scripts/train_screening.py
    --lib 3
    --seed "${SEED}"
    --ref-seed "${REF_SEED}"
    --reference-kind rate_limited_random
    --episodes "${EPISODES}"
    --horizon 512
    --carry
    --carry-reset-blocks 4
    --start-epi 10
    --gamma 0.99
    --scale-low 0.8
    --tracking-loss abs
    --chamber-reserve-rwd-coeff 0.05
    --chamber-bound-rwd-coeff 0.10
    --warmup-physical-action 0.95
    --warmup-mode pink
    --warmup-pink-sigma 0.20
    --defer-updates-during-warmup
    --reset-after-warmup
    --exploration none
    --normalize-observation
    --action-history-len 10
    --caps-mode mean_action
    --caps-temporal-weight 0.5
    --caps-spatial-weight 0.1
    --caps-noise-std 0.0
    --caps-spatial-noise-normalized 0.01
    --caps-start-after-warmup
    --caps-ramp-episodes 100
    --learning-rate 0.0003
    --batch-size 128
    --hidden-dim 256
    --buffer-size 500000
    --tau 0.005
    --alpha 0.5
    --epoch 1
    --update-every 1
    --log-std-min -10
    --log-std-max 1
    --verbose-every 25
)

case "$1" in
    1)
        VARIANT="history_sac"
        LABEL="history_sac"
        ;;
    2)
        VARIANT="history_sac+caps"
        LABEL="history_caps2"
        ;;
    3)
        VARIANT="mpobs1_sac"
        LABEL="mpobs1_sac"
        ;;
    4)
        VARIANT="sac+mpobs1+caps"
        LABEL="mpobs1_caps2"
        ;;
    5)
        VARIANT="history_action_sac"
        LABEL="history_action_sac"
        ;;
    6)
        VARIANT="history_action_sac+caps"
        LABEL="history_action_caps2"
        ;;
    *)
        echo "experiment must be one of 1, 2, 3, 4, 5, 6" >&2
        exit 2
        ;;
esac

NAME="260805_lib3_512_${LABEL}_protocol2_seed${SEED}"
if [[ -e "pneu_rl/models/${NAME}" ]]; then
    echo "refusing to overwrite existing model: pneu_rl/models/${NAME}" >&2
    exit 3
fi

OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 \
"${PYTHON_BIN}" "${COMMON[@]}" --variant "${VARIANT}" --name "${NAME}"
