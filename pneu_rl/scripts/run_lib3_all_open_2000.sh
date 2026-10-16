#!/usr/bin/env bash
set -euo pipefail
all_open_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$all_open_root"
all_open_python="${PYTHON_BIN:-/home/rise/miniconda3/envs/env_1/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/brl_all_open_training_mpl"
for policy in H HR; do
    model="261015_lib3_allopen30_${policy}_smooth_seed0"
    if [[ -e "pneu_rl/models/$model" ]]; then
        echo "Existing model folder: pneu_rl/models/$model; refusing to overwrite." >&2
        exit 1
    fi
done
mkdir -p pneu_rl/exp/lib3_all_open_training_logs "$MPLCONFIGDIR"
pids=()
for policy in H HR; do
    model="261015_lib3_allopen30_${policy}_smooth_seed0"
    "$all_open_python" -u pneu_rl/scripts/train_lib3_all_open.py \
        --policy "$policy" --episodes 2000 --precondition-seconds 30 \
        --threads 1 --name "$model" \
        > "pneu_rl/exp/lib3_all_open_training_logs/${policy}.log" 2>&1 &
    pids+=("$!")
    echo "$policy PID=$! log=pneu_rl/exp/lib3_all_open_training_logs/${policy}.log"
done
result=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then result=1; fi
done
exit "$result"
