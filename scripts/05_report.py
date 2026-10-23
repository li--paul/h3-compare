#!/usr/bin/env python3
"""Render a per-card report from <results>/timings.json.

Usage:
  python 05_report.py --results-dir results/rtxpro4500 \
      --output-dir output/rtxpro4500 --logs-dir logs/rtxpro4500
"""
import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arc Pro B60 single-card reference, from ~/Downloads/h3单卡测试总结_20261022.md
# REF2VA / 1376x768 / 124 frames / 8 steps / INT8 ConvRot, .118 GPU 31.
B60 = {
    "name": "Arc Pro B60 24 GB",
    "step": 70.0,
    "sampling": 563.0,
    "te": 16.4,
    "vae": 73.6,          # 653 - 563 - 16.4
    "segment": 653.0,     # 10m53s
    "peak_vram_gib": 20.2,
    "peak_power_w": 181.0,
    "avg_power_w": 148.0,
    "vram_gib": 24.0,
    "power_limit_w": 190.0,
    "source": "h3单卡测试总结_20261022.md",
}


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def gpu_name(logs_dir=None):
    """Prefer the card recorded at run time; fall back to the current GPU."""
    if logs_dir:
        card = os.path.join(logs_dir, "card.txt")
        if os.path.exists(card):
            line = open(card).read().strip().splitlines()[0]
            return [x.strip() for x in line.split(",")]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,power.limit",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        return [x.strip() for x in out.split(",")]
    except Exception:
        return ["unknown GPU", "", ""]


def gpu_stats(logs_dir):
    files = sorted(glob.glob(os.path.join(logs_dir, "nvidia-smi_*.csv")))
    if not files:
        return None
    rows = [r for r in csv.reader(open(files[-1])) if len(r) >= 6]
    if not rows:
        return None
    mem = [float(r[2]) for r in rows]
    pw = [float(r[3]) for r in rows]
    util = [float(r[1]) for r in rows]
    clk = [float(r[4]) for r in rows]
    busy = [u for u in util if u > 50]
    return {
        "peak_mem_mib": max(mem),
        "peak_power_w": max(pw),
        "mean_power_w": sum(pw) / len(pw),
        "mean_util_busy": sum(busy) / len(busy) if busy else 0.0,
        "busy_samples": len(busy),
        "samples": len(rows),
        "sm_clk_max": max(clk),
        "sm_clk_min": min(clk),
        "file": os.path.basename(files[-1]),
    }


def probe(path):
    if not path or not os.path.exists(path):
        return None
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
           "stream=codec_type,codec_name,width,height,nb_read_frames,r_frame_rate,sample_rate,channels"
           ":format=duration,size", "-of", "json", path]
    try:
        d = json.loads(subprocess.check_output(cmd, text=True))
    except Exception:
        return None
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "w": v.get("width"), "h": v.get("height"), "frames": v.get("nb_read_frames"),
        "fps": v.get("r_frame_rate"), "vcodec": v.get("codec_name"),
        "acodec": a.get("codec_name"), "sample_rate": a.get("sample_rate"),
        "channels": a.get("channels"),
        "duration": float(d.get("format", {}).get("duration", 0) or 0),
        "size_mb": float(d.get("format", {}).get("size", 0) or 0) / 2**20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS", os.path.join(ROOT, "results")))
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "output")))
    ap.add_argument("--logs-dir", default=os.environ.get("LOGS", os.path.join(ROOT, "logs")))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.results_dir, "timings.json"), encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    ok = [r for r in runs if not r.get("error")]
    name, mem_total, power_limit = gpu_name(args.logs_dir)
    if not name or name == "unknown GPU":
        name = os.path.basename(os.path.dirname(args.results_dir)) or "GPU"
    title = args.title or f"MiniMax-H3 Ref2VA 1376×768 / 8 步 — {name}"

    # CSV
    cols = ["label", "seed", "total_compute_s", "conditioning_s", "sampling_total_s",
            "step_median_s", "step_min_s", "step_max_s", "vae_video_s", "vae_audio_s",
            "save_s", "wall_s", "output"]
    with open(os.path.join(args.results_dir, "timings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            st = r.get("stages_s", {})
            steps = [s["s"] for s in r.get("steps", [])]
            w.writerow([
                r.get("label"), r.get("seed"), fmt(r.get("total_compute_s"), 2),
                fmt(st.get("conditioning_te_ref_vae"), 2), fmt(r.get("sampling_total_s"), 2),
                fmt(r.get("step_median_s"), 2),
                fmt(min(steps) if steps else None, 2), fmt(max(steps) if steps else None, 2),
                fmt(st.get("vae_decode_video"), 2), fmt(st.get("vae_decode_audio"), 2),
                fmt(st.get("save_video"), 2), fmt(r.get("wall_s"), 2), r.get("output") or "",
            ])

    lines = []
    A = lines.append
    A(f"# {title}")
    A("")
    A("由 `scripts/05_report.sh` 从 `timings.json` 自动生成；多卡对比见 `results/comparison.md`。")
    A("")
    A("## 配置与运行参数")
    A("")
    A("| 项 | 值 |")
    A("| --- | --- |")
    A("| 扩散主模型 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（20.97 GB） |")
    A("| 文本编码器 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（27.14 GB） |")
    A("| LoRA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |")
    A("| 分辨率 / 帧数 | 1376×768 / 124 帧（24 fps，约 5.17 s） |")
    A("| 采样 | res_multistep / beta / 8 步 / CFG 1.0 |")
    A("| 参考图 | `input/ref_1376x768.png`，`ref_image_size=match` |")
    A(f"| 硬件 | {name}，{mem_total} MiB，功耗上限 {power_limit} W |")
    A("| ComfyUI | 0.40.0（XPU 侧为 0.39.0），comfy-kitchen 0.2.37 |")
    flags = os.path.join(args.logs_dir, "run_flags.txt")
    if os.path.exists(flags):
        A(f"| 启动参数 | `{open(flags).read().strip().replace(chr(10), '; ')}` |")
    A("")
    A("## 分段耗时（秒）")
    A("")
    A("| run | seed | 条件编码(TE+参考VAE) | 采样合计 | 单步中位数 | 单步范围 | 视频 VAE | 音频 VAE | 总计算 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in runs:
        st = r.get("stages_s", {})
        steps = [s["s"] for s in r.get("steps", [])]
        A(f"| {r.get('label')} | {r.get('seed')} | {fmt(st.get('conditioning_te_ref_vae'))} | "
          f"{fmt(r.get('sampling_total_s'))} | {fmt(r.get('step_median_s'))} | "
          f"{fmt(min(steps) if steps else None)}–{fmt(max(steps) if steps else None)} | "
          f"{fmt(st.get('vae_decode_video'))} | {fmt(st.get('vae_decode_audio'))} | "
          f"{fmt(r.get('total_compute_s'))} |")
    A("")

    if ok:
        med_step = statistics.median(r["step_median_s"] for r in ok if r.get("step_median_s"))
        med_sampling = statistics.median(r["sampling_total_s"] for r in ok if r.get("sampling_total_s"))
        med_seg = statistics.median(r["total_compute_s"] for r in ok if r.get("total_compute_s"))
        med_vae = statistics.median(
            r["stages_s"].get("vae_decode_video", 0) + r["stages_s"].get("vae_decode_audio", 0)
            for r in ok)
        A("## 与 Arc Pro B60 单卡对照")
        A("")
        A("B60 数据来自 `h3单卡测试总结_20261022.md`（REF2VA / 1376×768 / 124 帧 / 8 步 / INT8）。")
        A("")
        A("| 指标 | B60 | 本机中位数 | 本机/XPU（耗时比） |")
        A("| --- | --- | --- | --- |")
        A(f"| 单步耗时 | {B60['step']:.0f} s | {med_step:.1f} s | {med_step/B60['step']:.2f}× |")
        A(f"| 8 步采样 | {B60['sampling']:.0f} s | {med_sampling:.1f} s | {med_sampling/B60['sampling']:.2f}× |")
        A(f"| 条件编码 | {B60['te']:.1f} s | "
          f"{statistics.median(r['stages_s'].get('conditioning_te_ref_vae', 0) for r in ok):.1f} s | — |")
        A(f"| VAE 解码 | {B60['vae']:.1f} s | {med_vae:.1f} s | {med_vae/B60['vae']:.2f}× |")
        A(f"| 单段总计 | {B60['segment']:.0f} s | {med_seg:.1f} s | {med_seg/B60['segment']:.2f}× |")
        A("")
        A(f"- 按单步计，本机比 B60 快 **{B60['step']/med_step:.1f} 倍**。")
        A(f"- 按 8 步采样计，快 **{B60['sampling']/med_sampling:.1f} 倍**。")
        A(f"- 按单段总计计，快 **{B60['segment']/med_seg:.1f} 倍**。")
        A("")

    gpu = gpu_stats(args.logs_dir)
    if gpu:
        A("## 实测运行状态（1 Hz 采样）")
        A("")
        A("| 指标 | 值 |")
        A("| --- | --- |")
        total_mib = int(mem_total) if str(mem_total).isdigit() else 0
        pct = f"（{gpu['peak_mem_mib']/total_mib*100:.0f}%）" if total_mib else ""
        A(f"| 峰值显存 | {gpu['peak_mem_mib']:.0f} MiB / {mem_total} MiB{pct} |")
        A(f"| 峰值功耗 | {gpu['peak_power_w']:.0f} W / {power_limit} W |")
        A(f"| 平均功耗 | {gpu['mean_power_w']:.0f} W |")
        A(f"| GPU 利用率（>50% 样本） | 均值 {gpu['mean_util_busy']:.0f}%（{gpu['busy_samples']}/{gpu['samples']} 样本） |")
        A(f"| SM 时钟范围 | {gpu['sm_clk_min']:.0f}–{gpu['sm_clk_max']:.0f} MHz |")
        A(f"| 原始数据 | `{os.path.relpath(os.path.join(args.logs_dir, gpu['file']), ROOT)}` |")
        A("")

    A("## 输出校验")
    A("")
    A("| 文件 | 视频 | 帧数 / fps | 音频 | 时长 | 大小 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for r in runs:
        p = probe(r.get("output"))
        if not p:
            continue
        A(f"| `{os.path.relpath(r['output'], ROOT)}` | {p['vcodec']} {p['w']}×{p['h']} | "
          f"{p['frames']} / {p['fps']} | {p['acodec']} {p['sample_rate']} Hz {p['channels']}ch | "
          f"{p['duration']:.2f} s | {p['size_mb']:.1f} MB |")
    A("")

    A("## 口径与限制")
    A("")
    A("- **权重逐字节一致**：与 XPU 交付包同名的 INT8 ConvRot 文件，字节数与 `results/sha256.txt` 已核对。")
    A("- **算子路径不同**：CUDA 上 `int8_tensorwise` 为 native INT8 GEMM；XPU 上该算子被禁用，权重反量化成 BF16。属于“同权重、不同 kernel”。")
    A("- **显存策略不同**：B60 为 24 GB + 流式换入；本机按实际显存容量驻留/换出，启动参数见上表。")
    A("- **输入差异**：B60 用《神笔马良》第 1 段提示词，本机用《草船借箭》提示词；token 数同量级，但非逐字节同输入。")
    A("- **版本差**：ComfyUI 0.40.0 vs 0.39.0；torch 2.14.1 基础版本一致；comfy-kitchen 均为 0.2.37。")
    A("- **缓存规避**：每次 run 更换 seed（1001/1002/1003）并把提示词 `[Shot 1]` 改为 `[Shot N]`。")
    A("")

    path = os.path.join(args.results_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
