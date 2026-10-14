#!/usr/bin/env bash
set -euo pipefail

# Six lib2 history/action ablations under the historical 2048-step reset
# protocol.  The implementation is kept in the original script to avoid
# duplicating the experiment definitions.
exec bash /home/rise/test/brl_window_v2/docs/ablation/lib2/run_5_history_rate_modules_h2048_reset.sh "$@"
