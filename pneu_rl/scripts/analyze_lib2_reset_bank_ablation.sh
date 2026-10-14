#!/usr/bin/env bash
# Distribution-level companion to analyze_lib2_reference_ablation.sh.
# It evaluates each finished reset variant over all ten v2 bank members, then
# compares per-model median/p90/worst tracking error.
set -euo pipefail

SEED="${1:-0}"
RUN_PREFIX="${RUN_PREFIX:-lib2_optref}"
TAG="${TAG:-reset_bank_seed${SEED}}"
MEMBERS="${BANK_MEMBERS:-10}"
EVAL_STEPS="${EVAL_STEPS:-2048}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VARIANTS=(sac sac+caps sac+mpobs1+caps sac+mpobs1+caps+iterm)
SUMMARY_ARGS=()
for VARIANT in "${VARIANTS[@]}"; do
  VARIANT_NAME="${VARIANT//+/\_}"
  MODEL_NAME="${RUN_PREFIX}_reset_${VARIANT_NAME}_h2048_seed${SEED}"
  BANK_MEMBERS="$MEMBERS" EVAL_STEPS="$EVAL_STEPS" \
    bash pneu_rl/scripts/evaluate_lib2_reset_reference_bank.sh "$MODEL_NAME" "$TAG"
  SUMMARY_ARGS+=(
    --bank-summary "pneu_rl/exp/_analysis/${MODEL_NAME}_${TAG}_bank/summary.yaml"
  )
done

MPLCONFIGDIR=/tmp/mpl PYTHONPATH= python3 pneu_rl/scripts/summarize_lib2_reference_bank.py \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "pneu_rl/exp/_analysis/${RUN_PREFIX}_reset_${TAG}_aggregate"
