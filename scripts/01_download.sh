#!/usr/bin/env bash
# Fetch the exact XPU artifacts (INT8 ConvRot) plus the 8-step Ref2VA LoRA.
# Idempotent: files with the expected size are skipped by hf itself.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "01_download"

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

DIT="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TE="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
LORA="minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
LORA_REPO="RunningHubAI/rh-minimax-h3-ref2v-turbo-8step-v1.0-768p-comfyui-bf16.safetensors-lora"

# Run the three transfers concurrently; each writes its own log.
hf download Comfy-Org/MiniMax-H3 "$DIT" --local-dir "$MODELS" > "$LOGS/download_dit.log" 2>&1 &
P1=$!
hf download Comfy-Org/MiniMax-H3 "$TE" --local-dir "$MODELS" > "$LOGS/download_te.log" 2>&1 &
P2=$!
hf download "$LORA_REPO" "$LORA" --local-dir "$MODELS/loras" > "$LOGS/download_lora.log" 2>&1 &
P3=$!
wait "$P1"; wait "$P2"; wait "$P3"

# Reuse the already-verified local VAEs (byte-identical to the Comfy-Org files).
cp -n /home/acm/work/models/h3-comfy/vae/minimax_h3_video_vae_fp16.safetensors "$MODELS/vae/" || true
cp -n /home/acm/work/models/h3-comfy/vae/minimax_h3_audio_vae_fp32.safetensors "$MODELS/vae/" || true

check() {
    local path="$1" want="$2" got
    got=$(stat -c %s "$path" 2>/dev/null || echo 0)
    if [[ "$got" != "$want" ]]; then
        echo "SIZE MISMATCH: $path got=$got want=$want" >&2
        return 1
    fi
    echo "OK $path ($got bytes)"
}

check "$MODELS/$DIT" 20970379616
check "$MODELS/$TE" 27141342152
check "$MODELS/loras/$LORA" 1956193000
check "$MODELS/vae/minimax_h3_video_vae_fp16.safetensors" 5207808496
check "$MODELS/vae/minimax_h3_audio_vae_fp32.safetensors" 605254808

( cd "$MODELS" && sha256sum "$DIT" "$TE" "loras/$LORA" \
    "vae/minimax_h3_video_vae_fp16.safetensors" "vae/minimax_h3_audio_vae_fp32.safetensors" ) \
    > "$ROOT/results/sha256.txt"

du -sh "$MODELS"
echo "download phase complete"
