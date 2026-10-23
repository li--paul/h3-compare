#!/usr/bin/env bash
# Start ComfyUI with every data path pointed at this benchmark directory.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "03_serve (RUN_TAG=$RUN_TAG)"

# Memory flags are the only permitted runtime difference between cards.
# Override with EXTRA_ARGS="--cache-none" / "--lowvram" when 32 GB is too small.
EXTRA_ARGS="${EXTRA_ARGS:-}"

if curl -sf "$BASE/system_stats" >/dev/null 2>&1; then
    echo "server already up on $BASE"
    exit 0
fi

printf '%s\n' "run_tag=$RUN_TAG" "port=$PORT" "extra_args=$EXTRA_ARGS" \
    "started=$(date -Is)" > "$LOGS/run_flags.txt"
nvidia-smi --query-gpu=name,memory.total,power.limit \
    --format=csv,noheader,nounits > "$LOGS/card.txt"
record_cmd "serve" "RUN_TAG=$RUN_TAG PORT=$PORT EXTRA_ARGS=$EXTRA_ARGS"

cd "$COMFY_DIR"
nohup "$VENV_PY" main.py \
    --listen 127.0.0.1 --port "$PORT" \
    --bf16-vae --preview-method none --disable-auto-launch \
    --extra-model-paths-config "$CONFIG" \
    --input-directory "$INPUT_DIR" \
    --output-directory "$OUTPUT_DIR" \
    --temp-directory "$TMP_DIR" \
    $EXTRA_ARGS \
    >> "$SERVER_LOG" 2>&1 &
echo $! > "$LOGS/server.pid"

wait_http "$BASE/system_stats" 900
echo "server ready: pid=$(cat "$LOGS/server.pid") log=$SERVER_LOG"
curl -s "$BASE/system_stats" | head -c 600; echo
