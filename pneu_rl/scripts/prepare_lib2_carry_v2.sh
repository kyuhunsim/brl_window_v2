#!/usr/bin/env bash
# Build and verify the *reference only* for the lib2 512-step carry ablation.
# This script intentionally does not start SAC training.  It makes the full
# 1000x512 simulator rollout reproducible, then requires carry continuity,
# oracle replay, and a fixed PID diagnostic before the profile is used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:-carry_v2_1000x512_seed743}"
SEED="${SEED:-743}"
PID_STEPS="${PID_STEPS:-512000}"
RUN_PID="${RUN_PID:-1}"
PROFILE="pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_1000x512.yaml"
NPZ="pneu_rl/config/reference/lib2_stabilized_carry_target_hold_v2_1000x512.npz"
PLOT="pneu_rl/img/reference/lib2_stabilized_carry_target_hold_v2_1000x512.png"
ANALYSIS="pneu_rl/exp/_analysis/lib2_optimized_target_hold_reference_v2/${TAG}"

mkdir -p "$ANALYSIS"

PYTHONPATH= MPLCONFIGDIR=/tmp/mpl_lib2ref python3 pneu_rl/scripts/generate_lib2_stabilized_carry_reference.py \
  --seed "$SEED" --duration 10240 --freq 50 --target-mode segmented \
  --target-pos-range 120,205 --target-neg-range 14,28 \
  --kp-pos 0.008 --kp-neg 0.013 --dither-amp-pos 0.04 --dither-amp-neg 0.03 \
  --pos-pressure-guard-kpa 205 --pos-pressure-guard-kp 0.03 --pos-pressure-guard-min-raw 0.38 \
  --progress-every-horizons 10 \
  --output "$PROFILE" --npz-output "$NPZ" --plot-output "$PLOT"

PYTHONPATH= python3 pneu_rl/scripts/validate_carry_reference.py \
  --reference-profile "$PROFILE" \
  --output "$ANALYSIS/carry_validation.yaml"

PYTHONPATH= python3 pneu_rl/scripts/check_lib2_carry_boundary.py \
  --reference-profile "$PROFILE" \
  --output "$ANALYSIS/carry_boundary_512.yaml" \
  --steps 512

PYTHONPATH= MPLCONFIGDIR=/tmp/mpl_lib2ref python3 pneu_rl/scripts/validate_optimized_reference.py \
  --reference-profile "$PROFILE" \
  --output-dir "$ANALYSIS/oracle_replay_target_start" \
  --init-pos 150 --init-neg 24

if [[ "$RUN_PID" == "1" ]]; then
  PYTHONPATH= MPLCONFIGDIR=/tmp/mpl_lib2ref python3 pneu_rl/scripts/evaluate_pid_reference_bank.py \
    --reference-profile "$PROFILE" \
    --tag "$TAG" \
    --steps "$PID_STEPS" \
    --freq 50 --delay 0.1 \
    --init-pos 101.325 --init-neg 101.325 \
    --base-pos 0.5 --base-neg 0.5 \
    --scale-low 0.85 --ki-pos 0.01 --ki-neg 0.01 --ka 10 \
    --ignore-initial-sec 5
fi

echo "[ INFO] carry v2 reference is ready for review: $ANALYSIS"
echo "[ INFO] inspect $PLOT before launching any carry SAC ablation."
