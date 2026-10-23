#!/usr/bin/env bash
# Render <RUN_TAG>/report.md + timings.csv from <RUN_TAG>/timings.json.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "05_report (RUN_TAG=$RUN_TAG)"
"$VENV_PY" "$ROOT/scripts/05_report.py" \
    --results-dir "$RESULTS" --output-dir "$OUTPUT_DIR" --logs-dir "$LOGS"
echo "report phase complete: $RESULTS/report.md"
