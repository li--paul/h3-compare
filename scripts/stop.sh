#!/usr/bin/env bash
# Stop the ComfyUI server started by 03_serve.sh for the given RUN_TAG.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
PID_FILE="$LOGS/server.pid"
if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "stopping ComfyUI pid=$PID (RUN_TAG=$RUN_TAG)" | tee -a "$COMMANDS_LOG"
        kill "$PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi
