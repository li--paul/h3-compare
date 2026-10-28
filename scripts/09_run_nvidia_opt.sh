#!/usr/bin/env bash
# Run the NVIDIA optimization matrix sequentially, one RUN_TAG per config.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

W6A8="$MODELS/diffusion_models/minimax_h3_ref2va_pruned_w6a8.safetensors"
FP8="$MODELS/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"

have_size() {
    local path="$1" want="$2"
    [[ -f "$path" && "$(stat -c %s "$path" 2>/dev/null || echo 0)" == "$want" ]]
}

have_sage() {
    "$VENV_PY" -c "import sageattention" >/dev/null 2>&1
}

run_cfg() {
    local tag="$1" wf="$2" extra="$3"
    echo "=================================================================="
    echo "=== config $tag  wf=$wf  extra='$extra'  $(date +%F\ %T)"
    echo "=================================================================="
    record_cmd "nvidia-opt" "RUN_TAG=$tag WORKFLOW=$wf EXTRA_ARGS=$extra"

    if [[ -f "$ROOT/results/$tag/timings.json" && "${FORCE:-0}" != "1" ]]; then
        echo "skip $tag: results already exist (FORCE=1 to rerun)"
        return 0
    fi

    if [[ "$tag" == *w6a8* ]] && ! have_size "$W6A8" 15983746636; then
        echo "skip $tag: W6A8 not complete"
        return 0
    fi
    if [[ "$tag" == *fp8* ]] && ! have_size "$FP8" 20958205608; then
        echo "skip $tag: FP8 not complete"
        return 0
    fi
    if [[ "$tag" == *sage* ]] && ! have_sage; then
        echo "skip $tag: sageattention not importable"
        return 0
    fi

    RUN_TAG="$tag" WORKFLOW="$wf" EXTRA_ARGS="$extra" bash "$ROOT/scripts/03_serve.sh" \
        || { echo "serve failed: $tag"; return 0; }
    RUN_TAG="$tag" WORKFLOW="$wf" bash "$ROOT/scripts/04_bench.sh" || echo "bench failed: $tag"
    RUN_TAG="$tag" bash "$ROOT/scripts/05_report.sh" || true
    RUN_TAG="$tag" bash "$ROOT/scripts/08_quality.sh" || true
    RUN_TAG="$tag" bash "$ROOT/scripts/stop.sh"
}

run_cfg rtxpro4500-sol        "$ROOT/workflows/ref2va_1376x768_8step_sol.json"   ""
run_cfg rtxpro4500-sol-tau10  "$ROOT/workflows/ref2va_1376x768_8step_sol_tau10.json" ""
run_cfg rtxpro4500-sol-tau20  "$ROOT/workflows/ref2va_1376x768_8step_sol_tau20.json" ""
run_cfg rtxpro4500-sol-fast   "$ROOT/workflows/ref2va_1376x768_8step_sol.json"   "--fast cublas_ops autotune"
run_cfg rtxpro4500-sol-sage   "$ROOT/workflows/ref2va_1376x768_8step_sol.json"   "--use-sage-attention"
run_cfg rtxpro4500-ckattn     "$ROOT/workflows/ref2va_1376x768_8step.json"       "--use-ck-attention"
run_cfg rtxpro4500-fast       "$ROOT/workflows/ref2va_1376x768_8step.json"       "--fast cublas_ops autotune"
run_cfg rtxpro4500-offload4   "$ROOT/workflows/ref2va_1376x768_8step.json"       "--async-offload 4"
run_cfg rtxpro4500-sage       "$ROOT/workflows/ref2va_1376x768_8step.json"       "--use-sage-attention"
run_cfg rtxpro4500-w6a8       "$ROOT/workflows/ref2va_1376x768_8step_w6a8.json"  ""
run_cfg rtxpro4500-fp8        "$ROOT/workflows/ref2va_1376x768_8step_fp8.json"   "--fast fp8_matrix_mult"

echo "=================================================================="
echo "=== all configs done  $(date +%F\ %T)"
"$VENV_PY" "$ROOT/scripts/07_nvidia_opt.py"
