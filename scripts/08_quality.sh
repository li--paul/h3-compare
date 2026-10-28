#!/usr/bin/env bash
# Compare one run's seed-1001 output against the INT8 baseline (PSNR/SSIM).
# Usage: bash scripts/08_quality.sh <tag>
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

TAG="${1:-$RUN_TAG}"
BASE="$ROOT/output/rtxpro4500/bench/h3_ref2va_1376x768_00001_.mp4"
CAND="$(find "$ROOT/output/$TAG" -name '*00001*.mp4' 2>/dev/null | head -1)"

if [[ ! -f "$BASE" || -z "$CAND" || ! -f "$CAND" ]]; then
    echo "quality: missing baseline or candidate (tag=$TAG)" >&2
    exit 0
fi

mkdir -p "$RESULTS"
PSNR="$(ffmpeg -hide_banner -loglevel info -i "$BASE" -i "$CAND" \
    -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -oE 'average:[0-9.]+' | tail -1 | cut -d: -f2)"
SSIM="$(ffmpeg -hide_banner -loglevel info -i "$BASE" -i "$CAND" \
    -lavfi "[0:v][1:v]ssim" -f null - 2>&1 | grep -oE 'All:[0-9.]+' | tail -1 | cut -d: -f2)"

{
    echo "psnr_avg=${PSNR:-}"
    echo "ssim_all=${SSIM:-}"
    echo "baseline=$BASE"
    echo "candidate=$CAND"
} > "$RESULTS/quality.txt"
cat "$RESULTS/quality.txt"
