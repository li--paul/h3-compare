#!/usr/bin/env python3
"""Build the multi-card comparison.

Same-host rows come from results/<tag>/timings.json + logs/<tag>/nvidia-smi_*.csv.
External rows come from results/sources/report-r9700-vs-b70.md (server A, 1344x768)
and are normalized to the same 1376x768 pixel count for per-step comparisons.
"""
import csv
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF_W, REF_H = 1376, 768
EXT_W, EXT_H = 1344, 768
NORM = (REF_W * REF_H) / (EXT_W * EXT_H)   # 1.0238
GPX = REF_W * REF_H * 124 / 1e9            # Gpixel per 124-frame segment

# Arc Pro B60 single-card reference, from results/sources/h3单卡测试总结_20261022.md
B60 = {
    "key": "b60",
    "name": "Intel Arc Pro B60",
    "vram": "24 GB",
    "power_limit": "—",
    "kernel": "INT8 ConvRot / XPU 反量化→BF16",
    "step": 70.0,
    "sampling": 563.0,
    "te": 16.4,
    "vae": 73.6,
    "segment": 653.0,
    "peak_vram_mib": 20.2 * 1024,
    "peak_power": 181.0,
    "avg_power": 148.0,
    "energy_power": 181.0,   # busy power (peak); avg includes load/idle
    "flags": "ComfyUI 0.39.0 + JZL 节点",
}

CARDS = [
    ("rtx6000ada", "NVIDIA RTX 6000 Ada"),
    ("rtxpro4500", "NVIDIA RTX PRO 4500 Blackwell"),
]

# Server A report (2026-10-23), 1344x768, INT8 ConvRot, 8 steps, seed 1001.
EXTERNAL = [
    {
        "name": "Intel Arc Pro B70", "host": "服务器 A", "res": "1344×768",
        "kernel": "INT8 ConvRot / XPU eager", "step": 40.29, "sampling": 322.3,
        "wall": 478.3, "peak_vram_gib": 28.4, "busy_power": 325.0,
        "note": "无 DynamicVRAM；仅 1 个有效跑次",
    },
    {
        "name": "AMD Radeon AI PRO R9700（默认）", "host": "服务器 A", "res": "1344×768",
        "kernel": "INT8 ConvRot / ROCm hip", "step": 44.92, "sampling": 359.4,
        "wall": 448.3, "peak_vram_gib": 28.1, "busy_power": 293.0,
        "note": "hip + DynamicVRAM（pinned 54.6 GB）",
    },
    {
        "name": "R9700 nodyn（关 DynamicVRAM）", "host": "服务器 A", "res": "1344×768",
        "kernel": "INT8 ConvRot / ROCm hip", "step": 44.97, "sampling": 359.8,
        "wall": 690.1, "peak_vram_gib": 28.7, "busy_power": 286.0,
        "note": "与 B70 同为旧式装载，墙钟被读盘拖累",
    },
    {
        "name": "R9700 ckt（triton 优先）", "host": "服务器 A", "res": "1344×768",
        "kernel": "INT8 ConvRot / ROCm triton", "step": 46.14, "sampling": 369.1,
        "wall": 509.5, "peak_vram_gib": 30.2, "busy_power": 289.0,
        "note": "COMFY_CK_PRIORITY=triton,hip,eager；整段反而更慢",
    },
]


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def load_runs(tag):
    path = os.path.join(ROOT, "results", tag, "timings.json")
    if not os.path.exists(path):
        return None
    runs = [r for r in json.load(open(path))["runs"] if not r.get("error")]
    if not runs:
        return None
    return {
        "step": med(r.get("step_median_s") for r in runs),
        "sampling": med(r.get("sampling_total_s") for r in runs),
        "te": med(r["stages_s"].get("conditioning_te_ref_vae") for r in runs),
        "vae": med(r["stages_s"].get("vae_decode_video", 0) + r["stages_s"].get("vae_decode_audio", 0)
                   for r in runs),
        "segment": med(r.get("total_compute_s") for r in runs),
        "runs": len(runs),
    }


def read_smi(tag):
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
    return {
        "peak_vram_mib": max(mem),
        "peak_power": max(pw),
        "avg_power": sum(busy) / len(busy) if busy else sum(pw) / len(pw),
    }


def read_text(tag, name):
    path = os.path.join(ROOT, "logs", tag, name)
    return open(path).read().strip() if os.path.exists(path) else ""


def card_row(key, name):
    if key == "b60":
        return dict(B60)
    runs = load_runs(key)
    if not runs:
        return None
    smi = read_smi(key)
    card = read_text(key, "card.txt").split(",")
    flags = read_text(key, "run_flags.txt")
    extra = ""
    for line in flags.splitlines():
        if line.startswith("extra_args="):
            extra = line.split("=", 1)[1].strip()
    return {
        "key": key,
        "name": card[0].strip() if card and card[0].strip() else name,
        "vram": f"{int(card[1]) / 1024:.1f} GB" if len(card) > 1 and card[1].strip().isdigit() else "—",
        "power_limit": f"{card[2].strip()} W" if len(card) > 2 else "—",
        "kernel": "INT8 ConvRot / CUDA INT8 GEMM",
        "step": runs["step"], "sampling": runs["sampling"], "te": runs["te"],
        "vae": runs["vae"], "segment": runs["segment"],
        "peak_vram_mib": smi.get("peak_vram_mib"),
        "peak_power": smi.get("peak_power"),
        "avg_power": smi.get("avg_power"),
        "energy_power": smi.get("avg_power"),   # busy average
        "flags": extra or "默认（--bf16-vae）",
    }


def f(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def kj(v):
    return "—" if v is None else f"{v/1000:.1f} kJ"


def ext_step(r):
    return r["step"] * NORM


def ext_energy(r):
    return ext_step(r) * r["busy_power"]


def main():
    rows = [dict(B60)]
    for key, name in CARDS:
        row = card_row(key, name)
        if row:
            rows.append(row)

    # ---------------------------------------------------------------- CSV
    out_csv = os.path.join(ROOT, "results", "comparison.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "gpu", "host", "resolution", "vram", "power_limit", "kernel",
                    "per_step_raw_s", "per_step_1376_s", "sampling_8step_s", "conditioning_s",
                    "vae_s", "segment_s", "wall_s", "peak_vram_mib", "peak_power_w",
                    "avg_power_w", "energy_per_step_j", "s_per_gpx", "segments_per_hour",
                    "wh_per_segment", "notes"])
        for r in rows:
            gpx = r["segment"] / GPX if r["segment"] else None
            sph = 3600 / r["segment"] if r["segment"] else None
            wh = (r["avg_power"] * r["segment"] / 3600) if r.get("avg_power") and r["segment"] else None
            energy = (r["step"] * r["energy_power"]) if r.get("energy_power") else None
            w.writerow(["same-host", r["name"], "本机", "1376x768", r["vram"], r["power_limit"],
                        r["kernel"], f(r["step"], 2), f(r["step"], 2), f(r["sampling"], 2),
                        f(r["te"], 2), f(r["vae"], 2), f(r["segment"], 2), f(r["segment"], 2),
                        f(r.get("peak_vram_mib"), 0), f(r.get("peak_power"), 1),
                        f(r.get("avg_power"), 1), f(energy, 0), f(gpx, 2), f(sph, 2),
                        f(wh, 2), r["flags"]])
        for r in EXTERNAL:
            w.writerow(["external", r["name"], r["host"], r["res"], "32 GB", "—", r["kernel"],
                        f(r["step"], 2), f(ext_step(r), 2), f(r["sampling"], 2), "", "",
                        "", f(r["wall"], 2), f(r["peak_vram_gib"] * 1024, 0), "",
                        "", f(ext_energy(r), 0), "", "", "", r["note"]])

    # ---------------------------------------------------------------- Markdown
    L = []
    A = L.append
    A("# MiniMax-H3 Ref2VA 单卡性能对比（同机 + 外部参考）")
    A("")
    A("统一条件：REF2VA / 124 帧 / 8 步 / CFG 1.0 / res_multistep+beta / batch 1 / INT8 ConvRot。")
    A("同机三行使用同一份输入、同一工作流、同一组权重（sha256 见 `results/sha256.txt`）。")
    A("")
    A("## 同机基准（1376×768）")
    A("")
    A("| 卡 | 显存 | 功耗上限 | 精度 / kernel | 单步@1376 | 8 步采样 | 条件编码 | VAE | 单段总计 | 峰值显存 | 功耗（峰/均） | 每步能耗 | 段能耗 | 段/小时 | s/Gpx |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        gpx = r["segment"] / GPX if r["segment"] else None
        sph = 3600 / r["segment"] if r["segment"] else None
        wh = (r["avg_power"] * r["segment"] / 3600) if r.get("avg_power") and r["segment"] else None
        energy = (r["step"] * r["energy_power"]) if r.get("energy_power") else None
        peak = f"{r['peak_vram_mib']/1024:.1f} GB" if r.get("peak_vram_mib") else "—"
        A(f"| {r['name']} | {r['vram']} | {r['power_limit']} | {r['kernel']} | "
          f"{f(r['step'])} s | {f(r['sampling'])} s | {f(r['te'])} s | {f(r['vae'])} s | "
          f"**{f(r['segment'])} s** | {peak} | {f(r.get('peak_power'))}/{f(r.get('avg_power'))} W | "
          f"{kj(energy)} | {f(wh, 1)} Wh | {f(sph, 2)} | {f(gpx, 2)} |")
    A("")
    A("> 每步能耗 = 单步 × 忙碌功耗；B60 用峰值 181 W，NVIDIA 用忙碌均值。")
    A("> 段能耗 = 平均功耗 × 单段，含装载/解码，仅同机三行可比。")
    A("")

    A("## 外部参考（服务器 A，1344×768）")
    A("")
    A("来源：`results/sources/report-r9700-vs-b70.md`（2026-10-23）。")
    A("")
    A("| 跑次 | 原始单步 | 单步@1376 | 8 步去噪 | 墙钟 | 显存峰值 | 忙碌功耗 | 每步能耗@1376 | 备注 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in EXTERNAL:
        A(f"| {r['name']} | {f(r['step'], 2)} s | {f(ext_step(r))} s | {f(r['sampling'])} s | "
          f"{f(r['wall'])} s | {f(r['peak_vram_gib'])} GB | {f(r['busy_power'])} W | "
          f"{kj(ext_energy(r))} | {r['note']} |")
    A("")
    A("- 归一化系数 `1376×768 / 1344×768 = 1.0238`；外部表的单步和每步能耗都乘了这个系数。")
    A("- `R9700 triton` 与默认实际同走 hip 后端（`--enable-triton-backend` 只加入候选、不改优先级），")
    A("  单步 44.95 s、墙钟 443.9 s，不单独成行。")
    A("- **墙钟不可跨主机比较**：服务器 A 是 61 GB 内存 + PCIe Gen4，本机 183 GB + Gen5；")
    A("  B70 还没有 XPU 版 DynamicVRAM。B70 478 s 与 R9700 默认 448 s 的差距主要来自装载方式，")
    A("  不是去噪算力；关掉 DynamicVRAM 后 R9700 变成 690 s（比 B70 慢 212 s）。")
    A("- 数据质量：B70 只有 1 个有效跑次；R9700 nodyn 成片亮度异常（110.1 vs 117.0）未查；")
    A("  R9700 默认跑次的利用率列是中途补采的。")
    A("")

    A("## 跨报告结论（归一化后）")
    A("")
    A("### 去噪算力")
    A("")
    A("| 排名 | 卡 | 单步@1376 | 耗时/B70（越小越快） | 来源 |")
    A("| --- | --- | --- | --- | --- |")
    ranked = []
    for r in rows:
        ranked.append((r["step"], r["name"], "同机"))
    for r in EXTERNAL:
        ranked.append((ext_step(r), r["name"], "服务器 A"))
    b70_step = ext_step(EXTERNAL[0])
    for i, (step, name, src) in enumerate(sorted(ranked), start=1):
        A(f"| {i} | {name} | {f(step)} s | {step/b70_step:.2f}× | {src} |")
    A("")
    A("- PRO 4500 与 Ada 基本持平（3% 以内）；两者比 B70 快约 1.76× / 1.72×。")
    A("- B70 比 R9700 默认快约 11%；R9700 nodyn 与默认去噪相同，慢在装载。")
    A("- B60 比 B70 慢 1.70×，比 PRO 4500 慢约 3.0×。")
    A("")
    A("### 每步能耗")
    A("")
    A("| 卡 | 每步能耗 | 能耗/B70（越小越省） |")
    A("| --- | --- | --- |")
    energies = [(r["step"] * r["energy_power"], r["name"]) for r in rows]
    energies += [(ext_energy(r), r["name"]) for r in EXTERNAL]
    b70_e = ext_energy(EXTERNAL[0])
    for e, name in sorted(energies):
        A(f"| {name} | {kj(e)} | {e/b70_e:.2f}× |")
    A("")
    A("- PRO 4500 每步能耗约为 B70 的 0.35×，Ada 约 0.53×，B60 约 0.95×。")
    A("- B70 与 R9700 默认几乎相同（13.4 vs 13.5 kJ/步）；R9700 ckt 略差。")
    A("")

    if len(rows) >= 2 and rows[0]["segment"]:
        A("## 同机相对 B60 的加速")
        A("")
        A("| 卡 | 单步 | 8 步采样 | 单段总计 | 段能耗（B60/本机） |")
        A("| --- | --- | --- | --- | --- |")
        base = rows[0]
        for r in rows[1:]:
            A(f"| {r['name']} | {base['step']/r['step']:.2f}× | "
              f"{base['sampling']/r['sampling']:.2f}× | {base['segment']/r['segment']:.2f}× | "
              f"{(base['avg_power']*base['segment'])/(r['avg_power']*r['segment']):.2f}× |")
        A("")

    ada = next((r for r in rows if r.get("key") == "rtx6000ada"), None)
    pro = next((r for r in rows if r.get("key") == "rtxpro4500"), None)
    if ada and pro:
        A("## RTX PRO 4500 vs RTX 6000 Ada")
        A("")
        step_ratio = pro["step"] / ada["step"]
        seg_ratio = pro["segment"] / ada["segment"]
        if step_ratio < 1:
            A(f"- 单步：PRO 4500 快 **{1/step_ratio:.2f}×**（{pro['step']:.1f} s vs {ada['step']:.1f} s）")
        else:
            A(f"- 单步：Ada 快 **{step_ratio:.2f}×**（{ada['step']:.1f} s vs {pro['step']:.1f} s）")
        if seg_ratio < 1:
            A(f"- 单段：PRO 4500 快 **{1/seg_ratio:.2f}×**（{pro['segment']:.1f} s vs {ada['segment']:.1f} s）")
        else:
            A(f"- 单段：Ada 快 **{seg_ratio:.2f}×**（{ada['segment']:.1f} s vs {pro['segment']:.1f} s）")
        A(f"- 峰值显存：Ada {ada['peak_vram_mib']/1024:.1f} GB vs PRO 4500 "
          f"{pro['peak_vram_mib']/1024:.1f} GB")
        A(f"- 段能耗：Ada {(ada['avg_power']*ada['segment'])/3600:.1f} Wh vs PRO 4500 "
          f"{(pro['avg_power']*pro['segment'])/3600:.1f} Wh "
          f"（PRO 4500 省 {(ada['avg_power']*ada['segment'])/(pro['avg_power']*pro['segment']):.2f}×）")
        A("")
        A("两者单步差距在 3% 以内，已在轮次波动范围内，应视为**基本持平**；")
        A("PRO 4500 用 2/3 的功耗和 2/3 的显存达到同样速度。")
        A("")

    A("## 口径说明")
    A("")
    A("- **B60 数据**：`results/sources/h3单卡测试总结_20261022.md`，.118 机器单卡 REF2VA；")
    A("  提示词为《神笔马良》第 1 段，与 NVIDIA 两轮的《草船借箭》不是逐字节同输入。")
    A("- **同权重、不同 kernel**：CUDA 上 `int8_tensorwise` 是 native INT8 GEMM；Intel XPU 上该算子被禁用，")
    A("  权重反量化后以 BF16 计算；R9700 走 ROCm hip。加速比不应全部归因于硬件。")
    A("- **Blackwell 原生算子**：PRO 4500 上 `int8_tensorwise`、`convrot_w4a4`、`nvfp4`、`mxfp8` 均为 native；")
    A("  Ada 上 `nvfp4` / `mxfp8` 为 emulated。本轮两边都只用 INT8，未使用 FP4。")
    A("- **显存策略不同**：B60 24 GB 流式换入；NVIDIA 两轮都启用 comfy-aimdo 0.5.6 动态显存——")
    A("  Ada 48 GB 可全驻留（峰值 48.3 GB），PRO 4500 32 GB 在 31.1 GB 峰值下自动换入换出，但未 OOM、未降级。")
    A("  实际启动参数写在 `logs/<tag>/run_flags.txt`（本轮为空，即默认 `--bf16-vae`）。")
    A("- **软件版本**：B60 为 ComfyUI 0.39.0 + JZL 节点；NVIDIA 为 0.40.0 + 核心节点；")
    A("  comfy-kitchen 均为 0.2.37。服务器 A 报告为 comfy-kitchen 0.2.39，B70 走 XPU eager，R9700 走 hip。")
    A("- **输出一致性**：同 seed 1001 下 Ada 与 PRO 4500 的画面结构一致、细节不同")
    A("  （PSNR≈19.7 dB，SSIM≈0.79），属于不同 INT8 kernel 的数值分歧；不要用逐像素一致做验收。")
    A("")

    path = os.path.join(ROOT, "results", "comparison.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
