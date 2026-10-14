#!/usr/bin/env bash
# Evaluate one completed reset-model across every member of the v2 reference
# bank.  This is distribution-level evaluation: one convenient trajectory
# must not stand in for the ten trajectories used during training.
set -euo pipefail

MODEL_NAME="${1:-}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "usage: evaluate_lib2_reset_reference_bank.sh MODEL_NAME [tag]" >&2
  exit 2
fi

TAG="${2:-reset_bank}"
PROFILE="${REFERENCE_PROFILE:-pneu_rl/config/reference/lib2_optimized_target_hold_bank_v2.yaml}"
MEMBERS="${BANK_MEMBERS:-10}"
EVAL_STEPS="${EVAL_STEPS:-2048}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f "pneu_rl/models/${MODEL_NAME}/cfg.yaml" ]]; then
  echo "missing model cfg: pneu_rl/models/${MODEL_NAME}/cfg.yaml" >&2
  exit 3
fi
if [[ "$MEMBERS" -lt 1 ]]; then
  echo "BANK_MEMBERS must be positive" >&2
  exit 4
fi

SCREEN_ARGS=()
for ((member=0; member<MEMBERS; member++)); do
  MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/evaluate_screening.py \
    --model "$MODEL_NAME" \
    --steps "$EVAL_STEPS" \
    --tag "$TAG" \
    --bank-member "$member" \
    --require-reference-match \
    --reference-profile "$PROFILE"
  SCREEN_ARGS+=(
    --screen-dir "pneu_rl/exp/${MODEL_NAME}/screen_eval_${TAG}_member$(printf '%02d' "$member")_${EVAL_STEPS}step"
  )
done

MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/summarize_screening_batch.py \
  "${SCREEN_ARGS[@]}" \
  --output-dir "pneu_rl/exp/_analysis/${MODEL_NAME}_${TAG}_bank"

echo "[ INFO] reset-bank evaluation complete: pneu_rl/exp/_analysis/${MODEL_NAME}_${TAG}_bank"
