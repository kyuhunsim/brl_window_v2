#!/usr/bin/env bash
set -euo pipefail

# Run one group per terminal.  Each terminal trains two models sequentially:
#   bash docs/ablation/lib4/run_6_screening_groups.sh 1
#   bash docs/ablation/lib4/run_6_screening_groups.sh 2
#   bash docs/ablation/lib4/run_6_screening_groups.sh 3

GROUP="${1:-}"
if [[ "$GROUP" != "1" && "$GROUP" != "2" && "$GROUP" != "3" ]]; then
    echo "usage: bash $0 {1|2|3}" >&2
    exit 2
fi

ROOT=/home/rise/test/brl_window_v2
PYTHON=/home/rise/miniconda3/envs/env_1/bin/python
PROFILE="$ROOT/pneu_rl/config/reference/lib4_capacity_core_16x4x512_seed64001.yaml"

cd "$ROOT"
mkdir -p /tmp/mpl-cache-lib4

run_train() {
    local variant="$1"
    local name="$2"

    env -u PYTHONPATH \
        MPLCONFIGDIR=/tmp/mpl-cache-lib4 \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" pneu_rl/scripts/train_lib4_screening.py \
        --variant "$variant" \
        --reference-profile "$PROFILE" \
        --seed 0 \
        --episodes 1000 \
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
        --name "$name"
}

case "$GROUP" in
    1)
        run_train sac 260823_lib4_r3_direct_seed0
        run_train history_sac 260823_lib4_r3_history_seed0
        ;;
    2)
        run_train history_rate 260823_lib4_r3_history_rate_seed0
        run_train history_reserve 260823_lib4_r3_history_reserve_seed0
        ;;
    3)
        run_train history_rate_reserve 260823_lib4_r3_history_rate_reserve_seed0
        run_train history_rate_reserve_k2 260823_lib4_r3_history_rate_reserve_k2_seed0
        ;;
esac
