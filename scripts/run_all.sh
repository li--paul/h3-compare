#!/usr/bin/env bash
# One-shot orchestrator: download -> prepare -> serve -> bench -> report -> stop.
# Every child script appends its own timestamped command trace to logs/commands.log.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

echo "=================================================================="
echo "=== run_all  $(date -Is)  root=$ROOT"
echo "=== matrix: ${RUNS:-cold:${SEED},warm:$((SEED+1)),warm:$((SEED+2))}"
echo "=================================================================="
printf '%s\t%s\t%s\n' "$(date +%F\ %T)" "run_all" \
    "01_download.sh && 02_prepare.sh && 03_serve.sh && 04_bench.sh && 05_report.sh" \
    >> "$LOGS/commands.log"

bash "$ROOT/scripts/01_download.sh"
bash "$ROOT/scripts/02_prepare.sh"
bash "$ROOT/scripts/03_serve.sh"
bash "$ROOT/scripts/04_bench.sh"
bash "$ROOT/scripts/05_report.sh"
bash "$ROOT/scripts/06_compare.sh"
bash "$ROOT/scripts/stop.sh"

echo "all phases complete: $(date -Is)"
