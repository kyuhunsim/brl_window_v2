#!/usr/bin/env bash
# Evaluate one completed four-model ablation set using the exact profile used
# by training, then add local-action perturbation and a batch summary.
set -euo pipefail

MODE="${1:-}"
SEED="${2:-0}"
RUN_PREFIX="${RUN_PREFIX:-lib2_optref}"
if [[ "$MODE" != "reset" && "$MODE" != "carry" ]]; then
  echo "usage: analyze_lib2_reference_ablation.sh {reset|carry} [seed]" >&2
  exit 2
fi

if [[ "$MODE" == "reset" ]]; then
  HORIZON=2048
  REFERENCE_PROFILE="pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml"
else
  # A 2048-step evaluation is intentionally continuous: it covers four
  # 512-step bookkeeping horizons without introducing a reset.
  HORIZON=512
  REFERENCE_PROFILE="${CARRY_REFERENCE_PROFILE:-pneu_rl/config/reference/lib2_stabilized_carry_target_stress_1000x512.yaml}"
fi
EVAL_STEPS=2048
TAG="ablation_${MODE}_seed${SEED}"
RUN_QUICK_VIZ="${RUN_QUICK_VIZ:-1}"
VARIANTS=(sac sac+caps sac+mpobs1+caps sac+mpobs1+caps+iterm)
SCREEN_ARGS=()

for VARIANT in "${VARIANTS[@]}"; do
  VARIANT_NAME="${VARIANT//+/\_}"
  MODEL_NAME="${RUN_PREFIX}_${MODE}_${VARIANT_NAME}_h${HORIZON}_seed${SEED}"
  MODEL_CFG="pneu_rl/models/${MODEL_NAME}/cfg.yaml"
  if [[ ! -f "$MODEL_CFG" ]]; then
    echo "missing completed model: $MODEL_CFG" >&2
    exit 3
  fi
  MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/evaluate_screening.py \
    --model "$MODEL_NAME" \
    --steps "$EVAL_STEPS" \
    --tag "$TAG" \
    --require-reference-match \
    --reference-profile "$REFERENCE_PROFILE"
  SCREEN_DIR="pneu_rl/exp/${MODEL_NAME}/screen_eval_${TAG}_${EVAL_STEPS}step"
  MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/analyze_control_opportunity.py \
    --screen-dir "$SCREEN_DIR" \
    --transient-sec 2.0 \
    --tolerance 5.0
  MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/local_action_perturbation.py \
    --screen-dir "$SCREEN_DIR" \
    --horizon 25 \
    --stride 100 \
    --max-points 30 \
    --transient-sec 2.0 \
    --delta-physical 0.005
  if [[ "$RUN_QUICK_VIZ" == "1" ]]; then
    # This is a second, compact diagnostic plot.  Unlike historical
    # quick-viz usage, the profile is mandatory and its content hash must
    # match the model cfg; non-interactive flags keep batch ablations safe.
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl QUICK_VIZ_MODEL="$MODEL_NAME" \
    QUICK_VIZ_REFERENCE_PROFILE="$REFERENCE_PROFILE" QUICK_VIZ_OBS_MODE=sim \
    QUICK_VIZ_PID_MODE=off QUICK_VIZ_SAVE=1 QUICK_VIZ_SAVE_NAME="quick_viz_${TAG}" \
    PYTHONPATH= python3 pneu_rl/scripts/quick_viz.py
  fi
  SCREEN_ARGS+=(--screen-dir "$SCREEN_DIR")
done

MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/summarize_screening_batch.py \
  "${SCREEN_ARGS[@]}" \
  --output-dir "pneu_rl/exp/_analysis/${RUN_PREFIX}_${MODE}_seed${SEED}"
