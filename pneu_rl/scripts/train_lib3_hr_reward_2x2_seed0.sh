#!/usr/bin/env bash
# Row HR of the 2x2: history plus explicit pressure-rate observation,
# base then smooth reward.  Run this in one terminal, sequentially.

set -euo pipefail

ROOT=/home/rise/test/brl_window_v2
cd "$ROOT"

bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-arm HR_base 0
bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-arm HR_smooth 0
