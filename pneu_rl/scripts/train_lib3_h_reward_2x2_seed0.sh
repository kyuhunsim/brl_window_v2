#!/usr/bin/env bash
# Row H of the 2x2: history-only observation, base then smooth reward.
# Run this in one terminal.  It trains sequentially to avoid GPU contention.

set -euo pipefail

ROOT=/home/rise/test/brl_window_v2
cd "$ROOT"

bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-arm H_base 0
bash pneu_rl/scripts/run_lib3_history_rate_reward_2x2.sh train-arm H_smooth 0
