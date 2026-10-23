#!/usr/bin/env bash
# Render results/comparison.md + comparison.csv from all archived card runs.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "06_compare"
"$VENV_PY" "$ROOT/scripts/06_compare.py"
echo "comparison written: $ROOT/results/comparison.md"
