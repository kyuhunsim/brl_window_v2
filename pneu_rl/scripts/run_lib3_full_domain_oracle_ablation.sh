#!/usr/bin/env bash
# Run one lib3 4x512 full-domain oracle-reference experiment.
# Use one terminal per experiment; this script does not start screen/tmux/tail.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash $0 {1|2|3|4|5|6|7|8|9}" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/rise/miniconda3/envs/env_1/bin/python}"
EPISODES="${EPISODES:-1000}"
PROFILE="pneu_rl/config/reference/lib3_full_domain_oracle_hard_4x512_seed61098.yaml"

SEED=0
VARIANT="history_action_sac"
LABEL="history_action_base"
EXTRA=()

case "$1" in
    1)
        LABEL="history_action_base_seed0"
        ;;
    2)
        SEED=1
        LABEL="history_action_base_seed1"
        ;;
    3)
        SEED=2
        LABEL="history_action_base_seed2"
        ;;
    4)
        LABEL="history_action_hist5_seed0"
        EXTRA+=(--action-history-len 5)
        ;;
    5)
        LABEL="history_action_rate005_seed0"
        EXTRA+=(--rate-rwd-coeff 0.05)
        ;;
    6)
        LABEL="history_action_conflict1_seed0"
        EXTRA+=(--conflict-rwd-coeff 1.0)
        ;;
    7)
        LABEL="history_action_chbound1_seed0"
        EXTRA+=(--chamber-bound-rwd-coeff 1.0)
        ;;
    8)
        VARIANT="history_action_sac+caps"
        LABEL="history_action_caps_seed0"
        ;;
    9)
        VARIANT="sac+mpobs1+caps"
        LABEL="mpobs1_caps_seed0"
        ;;
    *)
        echo "experiment must be one of 1..9" >&2
        exit 2
        ;;
esac

NAME="260805_lib3_oracle4x512_${LABEL}"
if [[ -e "pneu_rl/models/${NAME}" ]]; then
    echo "refusing to overwrite existing model: pneu_rl/models/${NAME}" >&2
    exit 3
fi

COMMON=(
    pneu_rl/scripts/train_screening.py
    --lib 3
    --variant "${VARIANT}"
    --seed "${SEED}"
    --ref-seed 61098
    --reference-profile "${PROFILE}"
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
    --name "${NAME}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    COMMON+=(--dry-run)
fi

OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 \
"${PYTHON_BIN}" "${COMMON[@]}" "${EXTRA[@]}"
