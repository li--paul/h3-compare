#!/usr/bin/env bash
# Shared environment for the MiniMax-H3 CUDA-vs-XPU benchmark.
# Everything the benchmark needs lives under $ROOT.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

export COMFY_DIR="/home/acm/work/test/comfy-h3/ComfyUI"
export VENV_PY="/home/acm/work/test/comfy-h3/.venv/bin/python"
export PORT="${PORT:-8190}"
export BASE="http://127.0.0.1:${PORT}"
export TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

export MODELS="$ROOT/models"
export INPUT_DIR="$ROOT/input"
export TMP_DIR="$ROOT/tmp"
export WORKFLOW="${WORKFLOW:-$ROOT/workflows/ref2va_1376x768_8step.json}"
export CONFIG="$ROOT/config/extra_model_paths.yaml"

# Per-card isolation: each card writes into its own results/output/logs subtree.
export RUN_TAG="${RUN_TAG:-rtxpro4500}"
export RESULTS="$ROOT/results/$RUN_TAG"
export OUTPUT_DIR="$ROOT/output/$RUN_TAG"
export LOGS="$ROOT/logs/$RUN_TAG"
export COMMANDS_LOG="$ROOT/logs/commands.log"
export SERVER_LOG="$LOGS/comfy_${TS}.log"
export SMI_LOG="$LOGS/nvidia-smi_${TS}.csv"

# Bench geometry / sampling settings (must match the XPU reference run).
export WIDTH="${WIDTH:-1376}"
export HEIGHT="${HEIGHT:-768}"
export FRAMES="${FRAMES:-124}"
export STEPS="${STEPS:-8}"
export SEED="${SEED:-1001}"
export SAMPLER="${SAMPLER:-res_multistep}"
export SCHEDULER="${SCHEDULER:-beta}"
export PROMPT_FILE="$INPUT_DIR/prompt.txt"
export REF_IMAGE="ref_1376x768.png"

mkdir -p "$LOGS" "$RESULTS" "$OUTPUT_DIR" "$TMP_DIR" "$MODELS/vae" \
         "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/loras" "$INPUT_DIR"

# Record every command (with a timestamp) to logs/commands.log.
export PS4='+ [$(date +%F\ %T)] '
start_log() {
    local phase="${1:-phase}"
    mkdir -p "$LOGS"
    exec > >(tee -a "$COMMANDS_LOG") 2>&1
    echo "=================================================================="
    echo "=== ${phase}  host=$(hostname)  user=${USER}  $(date +%F\ %T)"
    echo "=== cwd=$(pwd)"
    echo "=================================================================="
    set -x
    trap 'echo "[$(date +%F\ %T)] FAILED rc=$? : $BASH_COMMAND" | tee -a "$COMMANDS_LOG" >&2' ERR
}

record_cmd() {  # record_cmd <description> <command...>
    local desc="$1"; shift
    printf '%s\t%s\t%s\n' "$(date +%F\ %T)" "$desc" "$*" >> "$COMMANDS_LOG"
}

wait_http() {
    local url="$1" tries="${2:-600}"
    for _ in $(seq 1 "$tries"); do
        if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "ERROR: timed out waiting for $url" >&2
    return 1
}
