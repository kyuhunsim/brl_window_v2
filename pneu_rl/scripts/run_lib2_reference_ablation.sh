#!/usr/bin/env bash
# Run exactly one lib2 reference-ablation job.  Invoke this script once per
# terminal so the host does not become overloaded by many C++ simulators.
set -euo pipefail

MODE="${1:-}"
VARIANT="${2:-}"
if [[ -z "$MODE" || -z "$VARIANT" ]]; then
  echo "usage: run_lib2_reference_ablation.sh {reset|carry} {sac|sac+caps|sac+mpobs1+caps|sac+mpobs1+caps+iterm} [seed]" >&2
  exit 2
fi
SEED="${3:-0}"
EPISODES="${EPISODES:-1000}"
RUN_PREFIX="${RUN_PREFIX:-lib2_optref}"

case "$MODE" in
  reset)
    HORIZON=2048
    # v2 is simulator-rolled and explicitly randomizes POS/NEG transition
    # count, transition duration, hold duration, and timing independently.
    REFERENCE_PROFILE="pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml"
    CARRY_ARGS=()
    ;;
  carry)
    HORIZON=512
    # Keep the legacy profile as a fallback.  Once the v2 1000x512 profile is
    # generated and validated, pass CARRY_REFERENCE_PROFILE=<that yaml>.
    REFERENCE_PROFILE="${CARRY_REFERENCE_PROFILE:-pneu_rl/config/reference/lib2_stabilized_carry_target_stress_1000x512.yaml}"
    CARRY_ARGS=(--carry)
    ;;
  *)
    echo "MODE must be reset or carry" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  sac|sac+caps|sac+mpobs1+caps|sac+mpobs1+caps+iterm) ;;
  *)
    echo "Unsupported ablation variant: $VARIANT" >&2
    exit 2
    ;;
esac

CAPS_ARGS=()
if [[ "$VARIANT" == *caps* ]]; then
  # Reduced CAPS weights: CAPS stays a policy-loss ablation, but cannot
  # dominate tracking in the first pure-RL comparison.
  CAPS_ARGS=(--caps-temporal-weight 0.10 --caps-spatial-weight 0.08 --caps-noise-std 1.0)
fi

VARIANT_NAME="${VARIANT//+/\_}"
MODEL_NAME="${RUN_PREFIX}_${MODE}_${VARIANT_NAME}_h${HORIZON}_seed${SEED}"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH= \
python3 pneu_rl/scripts/train_screening.py \
  --lib 2 \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --ref-seed 61098 \
  --episodes "$EPISODES" \
  --horizon "$HORIZON" \
  --start-epi 10 \
  --gamma 0.99 \
  --num-prev 10 \
  --scale-low 0.85 \
  --action-low-physical 0.90 \
  --warmup-physical-action 0.95 \
  --verbose-every 10 \
  --reference-independent-reset \
  --reference-profile "$REFERENCE_PROFILE" \
  "${CARRY_ARGS[@]}" \
  "${CAPS_ARGS[@]}" \
  --name "$MODEL_NAME"
