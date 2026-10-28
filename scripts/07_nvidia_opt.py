#!/usr/bin/env python3
"""Aggregate the NVIDIA H3 optimization runs into one report."""
import csv
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

CONFIGS = [
    ("rtxpro4500", "INT8 baseline", "same-quant", "PyTorch attention, default"),
    ("rtxpro4500-sol", "INT8 + SOL sparse tau=1.3", "same-quant", "BlockSparseAttention, tau=1.3, start=0.0"),
    ("rtxpro4500-sol-tau10", "INT8 + SOL tau=1.0", "same-quant", "less sparse, higher quality"),
    ("rtxpro4500-sol-tau20", "INT8 + SOL tau=2.0", "same-quant", "more sparse, faster/lower quality"),
    ("rtxpro4500-sol-fast", "INT8 + SOL + fast", "same-quant", "SOL + --fast cublas_ops autotune"),
    ("rtxpro4500-sol-sage", "INT8 + SOL + Sage2.2", "same-quant", "SOL + --use-sage-attention (compat check)"),
    ("rtxpro4500-ckattn", "INT8 + CK attention", "same-quant", "--use-ck-attention"),
    ("rtxpro4500-sage", "INT8 + SageAttention 2.2", "same-quant", "--use-sage-attention"),
    ("rtxpro4500-fast", "INT8 + fast(cublas/autotune)", "same-quant", "--fast cublas_ops autotune"),
    ("rtxpro4500-offload4", "INT8 + async-offload 4", "same-quant", "--async-offload 4"),
    ("rtxpro4500-w6a8", "W6A8 (official)", "quant-variant", "6-bit weights, int8 compute"),
    ("rtxpro4500-fp8", "FP8 scaled (official)", "quant-variant", "--fast fp8_matrix_mult"),
]


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def load_timings(tag):
    path = os.path.join(RESULTS, tag, "timings.json")
    if not os.path.exists(path):
        return None
    runs = [r for r in json.load(open(path))["runs"] if not r.get("error")]
    if not runs:
        return None
    return {
        "step": med(r.get("step_median_s") for r in runs),
        "sampling": med(r.get("sampling_total_s") for r in runs),
        "segment": med(r.get("total_compute_s") for r in runs),
        "te": med(r["stages_s"].get("conditioning_te_ref_vae") for r in runs),
        "vae": med(r["stages_s"].get("vae_decode_video", 0) + r["stages_s"].get("vae_decode_audio", 0)
                   for r in runs),
        "runs": len(runs),
    }


def load_smi(tag):
    files = sorted(glob.glob(os.path.join(ROOT, "logs", tag, "nvidia-smi_*.csv")))
    if not files:
        return {}
    rows = [r for r in csv.reader(open(files[-1])) if len(r) >= 6]
    if not rows:
        return {}
    mem = [float(r[2]) for r in rows]
    pw = [float(r[3]) for r in rows]
    util = [float(r[1]) for r in rows]
    busy = [p for p, u in zip(pw, util) if u > 50]
    return {"peak_vram_mib": max(mem), "peak_power": max(pw),
            "avg_power": sum(busy) / len(busy) if busy else sum(pw) / len(pw)}


def load_flags(tag):
    path = os.path.join(ROOT, "logs", tag, "run_flags.txt")
    if not os.path.exists(path):
        return ""
    for line in open(path):
        if line.startswith("extra_args="):
            return line.split("=", 1)[1].strip()
    return ""


def load_quality(tag):
    path = os.path.join(RESULTS, tag, "quality.txt")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            out[k] = v
    return out


def f(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    rows = []
    for tag, label, kind, note in CONFIGS:
        t = load_timings(tag)
        if not t:
            continue
        smi = load_smi(tag)
        q = load_quality(tag)
        rows.append({"tag": tag, "label": label, "kind": kind, "note": note,
                     "flags": load_flags(tag), "quality": q, **t, **smi})
    if not rows:
        print("no runs found")
        return 1

    base = rows[0]

    out_csv = os.path.join(RESULTS, "nvidia-optimizations.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "config", "kind", "per_step_s", "sampling_8step_s", "segment_s",
                    "speedup_vs_baseline", "peak_vram_mib", "peak_power_w", "avg_power_w",
                    "psnr_db", "ssim", "flags", "note"])
        for r in rows:
            w.writerow([r["tag"], r["label"], r["kind"], f(r["step"], 2), f(r["sampling"], 2),
                        f(r["segment"], 2), f(base["step"] / r["step"], 2),
                        f(r.get("peak_vram_mib"), 0), f(r.get("peak_power"), 1),
                        f(r.get("avg_power"), 1), r["quality"].get("psnr_avg", ""),
                        r["quality"].get("ssim_all", ""), r["flags"], r["note"]])

    L = []
    A = L.append
    A("# NVIDIA H3 / ComfyUI 优化对比（RTX PRO 4500 Blackwell 32 GB）")
    A("")
    A("统一条件：REF2VA / 1376×768 / 124 帧 / 8 步 / CFG 1.0 / res_multistep+beta / seed 1001–1003。")
    A("基线为 INT8 ConvRot + PyTorch attention；其余只改运行时/内核，或明确标注换量化。")
    A("")
    A("## 总表")
    A("")
    A("| 配置 | 类型 | 单步 | 8 步采样 | 端到端 | 峰值显存 | 功耗（峰/均） | 相对基线 | PSNR / SSIM | 备注 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        peak = f"{r['peak_vram_mib']/1024:.1f} GB" if r.get("peak_vram_mib") else "—"
        q = r["quality"]
        quality = f"{q.get('psnr_avg','—')} dB / {q.get('ssim_all','—')}" if q else "—"
        A(f"| {r['label']} | {r['kind']} | {f(r['step'],2)} s | {f(r['sampling'])} s | "
          f"**{f(r['segment'])} s** | {peak} | {f(r.get('peak_power'))}/{f(r.get('avg_power'))} W | "
          f"{f(base['step']/r['step'],2)}× | {quality} | {r['note']} |")
    A("")
    A("> 相对基线 = 基线单步 / 本配置单步；>1 表示更快。PSNR/SSIM 是对 seed 1001 输出与基线的逐帧比较。")
    A("> `same-quant` 行保持 INT8 ConvRot；`quant-variant` 行改变了量化格式，不能与 INT8 直接等同。")
    A("")

    A("## 分项")
    A("")
    for r in rows:
        A(f"- **{r['label']}**：{f(r['step'],2)} s/步，{f(r['sampling'])} s 采样，"
          f"{f(r['segment'])} s 端到端，{f(r['te'])} s 条件编码，{f(r['vae'])} s VAE。"
          + (f" 启动参数：`{r['flags']}`。" if r["flags"] else ""))
    A("")

    A("## 推荐")
    A("")
    A("- **同量化最快：SOL tau=1.3**，9.80 s/步 / 115.3 s，2.39×；PSNR 18.0 dB、SSIM 0.713，画面结构一致但采样轨迹不同。")
    A("- **画质优先：CK attention 或 SageAttention 2.2**，约 1.4×；CK 的 PSNR 23.1 dB / SSIM 0.814 更好，Sage 略快（16.42 vs 16.57 s/步）。")
    A("- **SOL tau=2.0 不推荐**：9.09 s/步 / 104.5 s 虽然最快，但 PSNR 14.3 dB / SSIM 0.600，构图已明显偏离。")
    A("- **SOL + Sage / SOL + fast 不叠加收益**：SOL 已接管 attention，额外 attention 后端不再加速；`--fast` 反而略慢。")
    A("- **W6A8 / FP8 在 32 GB 上更慢**：24.5 / 26.2 s/步；27 GB 文本编码器仍占满显存，换量化没有带来装载优势。")
    A("- **`--fast cublas_ops autotune` 不建议**：速度 1.01×，但同 seed 输出已变化（PSNR 21.8 dB），属于数值路径改变。")
    A("")

    A("## 口径与限制")
    A("")
    A("- 所有 `same-quant` 行使用同一份 INT8 ConvRot 权重；只有 kernel/attention/offload 变化。")
    A("- `W6A8` 是 6-bit 权重 + int8 计算，`FP8` 是 FP8 scaled；两者都改变了量化格式，画质需单独看 PSNR/抽帧。")
    A("- SOL 会改变采样轨迹（不是逐像素等价），PSNR 下降是预期的；看画面结构而不是 PSNR 绝对值。")
    A("- SageAttention 需要 2.2.0；PyPI 只有 1.0.6，本轮从 `thu-ml/SageAttention@v2.2.0` 源码编译。")
    A("- 32 GB 显存下 FP8/W6A8 的换入行为由 comfy-aimdo 决定，峰值显存和端到端时间受装载影响。")
    A("")

    path = os.path.join(RESULTS, "nvidia-optimizations.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
