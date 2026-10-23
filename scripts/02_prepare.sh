#!/usr/bin/env bash
# Build the local inputs (prompt + reference frame) and record environment facts.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
start_log "02_prepare"

SRC_VIDEO="/home/acm/work/test/ccj/final/caochuan_5min.mp4"
SRC_PROMPT="/home/acm/work/test/ccj/out/smoke_prompt.txt"

cp -f "$SRC_PROMPT" "$PROMPT_FILE"

# 1376x768 = 43:24; the source is 1920x1080, so scale up then center-crop.
ffmpeg -hide_banner -loglevel error -y -ss 60 -i "$SRC_VIDEO" -frames:v 1 \
    -vf "scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=increase,crop=${WIDTH}:${HEIGHT}" \
    "$INPUT_DIR/$REF_IMAGE"

"$VENV_PY" - "$INPUT_DIR/$REF_IMAGE" "$WIDTH" "$HEIGHT" <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1])
print("ref image:", im.size, im.mode)
want = (int(sys.argv[2]), int(sys.argv[3]))
assert im.size == want, f"{im.size} != {want}"
PY

cat > "$LOGS/env_${TS}.txt" <<EOF
date            $(date -Is)
host            $(hostname)
kernel          $(uname -r)
cpu             $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)
ram_gib         $(free -g | awk '/Mem:/{print $2}')
gpu             $(nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv,noheader)
torch           $("$VENV_PY" -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))')
comfy_commit    $(git -C "$COMFY_DIR" log -1 --format='%H %ci')
comfy_version   $(cat "$COMFY_DIR/comfyui_version.py" | grep __version__)
comfy_kitchen   $("$VENV_PY" -c 'import comfy_kitchen,importlib.metadata as m;print(m.version("comfy-kitchen"))' 2>/dev/null || echo unknown)
sampler         $SAMPLER / $SCHEDULER / steps=$STEPS / cfg=1.0
geometry        ${WIDTH}x${HEIGHT} / ${FRAMES} frames / seed base $SEED
prompt_sha256   $(sha256sum "$PROMPT_FILE" | cut -d' ' -f1)
ref_sha256      $(sha256sum "$INPUT_DIR/$REF_IMAGE" | cut -d' ' -f1)
EOF
cat "$LOGS/env_${TS}.txt"
echo "prepare phase complete"
