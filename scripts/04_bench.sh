#!/usr/bin/env bash
# Run the benchmark matrix against an already-running ComfyUI server.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "04_bench"

mkdir -p "$LOGS/events" "$OUTPUT_DIR/bench"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -l 1 > "$SMI_LOG" 2>&1 &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

RUNS="${RUNS:-cold:${SEED},warm:$((SEED+1)),warm:$((SEED+2))}"
record_cmd "bench matrix" "RUNS=$RUNS"

"$VENV_PY" "$ROOT/scripts/bench.py" \
    --base "$BASE" \
    --workflow "$WORKFLOW" \
    --prompt-file "$PROMPT_FILE" \
    --outdir "$OUTPUT_DIR" \
    --events-dir "$LOGS/events" \
    --results "$RESULTS/timings.json" \
    --runs "$RUNS" \
    2>&1 | tee "$LOGS/bench_console_${TS}.log"

# Keep a machine-readable stdout copy of the stage table.
kill "$SMI_PID" 2>/dev/null || true
trap - EXIT
echo "bench phase complete"
