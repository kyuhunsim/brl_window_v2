#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${REPO_ROOT}/pneu_env/src:${REPO_ROOT}/pneu_ref/src:${REPO_ROOT}/pneu_rl/src:${REPO_ROOT}/pneu_utils/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Configured PYTHONPATH for ${REPO_ROOT}"
