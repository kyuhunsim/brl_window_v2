#!/usr/bin/env bash
# Run exactly one member of the six-cell lib3 ablation matrix (IDs 3--8).
#
# The reference preflight is a separate command and must pass before using
# experiment 5 or 6.  The default horizon is 512 with carry so each 512-step
# episode is a learning boundary while the physical rollout remains continuous.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 {3|4|5|6|7|8}" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/rise/miniconda3/envs/env_1/bin/python}"
EPISODES="${EPISODES:-1000}"
HORIZON="${HORIZON:-512}"
SEED="${SEED:-0}"
REF_SEED="${REF_SEED:-61098}"

# Always regenerate the exact preflight before training.  This is cheap and
# prevents a stale reference/config from silently entering the six-cell run.
env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl-cache "$PYTHON_BIN" \
    pneu_rl/scripts/validate_lib3_reference.py \
    --ref-seed "${REF_SEED}" \
    --blocks 4 \
    --block-steps 512 \
    --chamber-controller dynamic_reserve \
    --output-dir pneu_rl/exp/lib3_reference_preflight/rate_limited_seed"${REF_SEED}"

COMMON=(
    pneu_rl/scripts/train_screening.py
    --lib 3
    --seed "${SEED}"
    --ref-seed "${REF_SEED}"
    --reference-kind rate_limited_random
    --episodes "${EPISODES}"
    --horizon "${HORIZON}"
    --carry
    --start-epi 10
    --gamma 0.99
    --scale-low 0.8
    --exploration none
)

case "$1" in
    3)
        EXTRA=(
            --variant sac
            --include-physics-features
            --name "lib3_512_6a_sac_physics_seed${SEED}"
        )
        ;;
    4)
        EXTRA=(
            --variant sac+mpobs1+caps
            --include-physics-features
            --name "lib3_512_6a_mpobs1_caps_physics_seed${SEED}"
        )
        ;;
    5)
        EXTRA=(
            --variant sac
            --chamber-controller dynamic_reserve
            --name "lib3_512_4a_chamber_sac_seed${SEED}"
        )
        ;;
    6)
        EXTRA=(
            --variant sac
            --chamber-controller dynamic_reserve
            --include-physics-features
            --name "lib3_512_4a_chamber_sac_physics_seed${SEED}"
        )
        ;;
    7)
        EXTRA=(
            --variant sac+mpobs1+caps
            --chamber-controller dynamic_reserve
            --name "lib3_512_4a_chamber_mpobs1_caps_seed${SEED}"
        )
        ;;
    8)
        EXTRA=(
            --variant sac+mpobs1+caps
            --chamber-controller dynamic_reserve
            --include-physics-features
            --name "lib3_512_4a_chamber_mpobs1_caps_physics_seed${SEED}"
        )
        ;;
    *)
        echo "experiment must be one of 3, 4, 5, 6, 7, 8" >&2
        exit 2
        ;;
esac

env -u PYTHONPATH OMP_NUM_THREADS=2 "$PYTHON_BIN" "${COMMON[@]}" "${EXTRA[@]}"
