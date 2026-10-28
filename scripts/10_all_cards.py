#!/usr/bin/env python3
"""One table across Intel / AMD / NVIDIA, curated representative configs."""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# step/e2e are at 1376x768. B70/R9700 were measured at 1344x768 and normalized.
ROWS = [
    dict(card="NVIDIA RTX PRO 4500 + SOL", type="NVIDIA 优化", vendor="NVIDIA",
         step=9.80, e2e=115.3, vram=31.3, peak=203.3, avg=195.9,
         psnr=18.01, ssim=0.713, note="同量化最快；tau=1.3；改变采样轨迹"),
    dict(card="NVIDIA RTX PRO 4500 + SageAttention 2.2", type="NVIDIA 优化", vendor="NVIDIA",
         step=16.42, e2e=161.1, vram=31.1, peak=203.6, avg=198.8,
         psnr=21.46, ssim=0.804, note="需源码编译 SageAttention 2.2.0"),
    dict(card="NVIDIA RTX PRO 4500 + CK attention", type="NVIDIA 优化", vendor="NVIDIA",
         step=16.57, e2e=162.3, vram=31.2, peak=203.2, avg=198.7,
         psnr=23.05, ssim=0.814, note="画质优先；PSNR 最接近基线"),
    dict(card="NVIDIA RTX PRO 4500（基线）", type="基线", vendor="NVIDIA",
         step=23.43, e2e=216.6, vram=31.1, peak=203.4, avg=198.0,
         psnr=None, ssim=None, note="PyTorch attention；32 GB aimdo 换入"),
    dict(card="NVIDIA RTX 6000 Ada", type="基线", vendor="NVIDIA",
         step=24.04, e2e=223.3, vram=47.2, peak=303.5, avg=297.3,
         psnr=None, ssim=None, note="48 GB 全驻留"),
    dict(card="Intel Arc Pro B60 + SOL", type="XPU 优化", vendor="Intel",
         step=36.97, e2e=370.2, vram=21.4, peak=None, avg=None,
         psnr=None, ssim=None, note="Intel llm-scaler 镜像；稀疏注意力"),
    dict(card="Intel Arc Pro B70", type="基线", vendor="Intel",
         step=41.25, e2e=478.3, vram=28.4, peak=325.0, avg=None,
         psnr=None, ssim=None, note="1344→1376 归一化；无 DynamicVRAM"),
    dict(card="AMD Radeon AI PRO R9700", type="基线", vendor="AMD",
         step=45.99, e2e=448.3, vram=28.1, peak=293.0, avg=None,
         psnr=None, ssim=None, note="1344→1376 归一化；开 DynamicVRAM"),
    dict(card="Intel Arc Pro B60 + Intel dense", type="XPU 优化", vendor="Intel",
         step=46.40, e2e=470.2, vram=21.4, peak=None, avg=None,
         psnr=None, ssim=None, note="Intel 镜像 dense；native INT8 + CUTE"),
    dict(card="Intel Arc Pro B60（stock）", type="基线", vendor="Intel",
         step=70.0, e2e=653.0, vram=20.2, peak=181.0, avg=148.0,
         psnr=None, ssim=None, note="交付包；INT8 仅存储、BF16 计算"),
]

BASE_STEP = 70.0


def f(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    rows = sorted(ROWS, key=lambda r: (r["step"], r["card"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["speedup"] = BASE_STEP / r["step"]

    out_csv = os.path.join(ROOT, "results", "all-cards.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "card_config", "type", "vendor", "per_step_1376_s", "e2e_s",
                    "peak_vram_gib", "peak_power_w", "avg_power_w", "speedup_vs_b60_stock",
                    "psnr_db", "ssim", "note"])
        for r in rows:
            w.writerow([r["rank"], r["card"], r["type"], r["vendor"], f(r["step"], 2),
                        f(r["e2e"], 2), f(r["vram"]), f(r["peak"]), f(r["avg"]),
                        f(r["speedup"], 2), f(r["psnr"], 2), f(r["ssim"], 3), r["note"]])

    L = []
    A = L.append
    A("# MiniMax-H3 全卡单段对比（Intel / AMD / NVIDIA）")
    A("")
    A("统一条件：REF2VA / 124 帧 / 8 步 / CFG 1.0 / res_multistep+beta / seed 1001–1003 / INT8 ConvRot（除特别注明）。")
    A("按单步@1376×768 排序；相对加速以 Intel B60 stock 的 70.0 s/步为 1.00×。")
    A("")
    A("| # | 卡 / 配置 | 类型 | 单步@1376 | 端到端 | 峰值显存 | 功耗（峰/均） | 相对 B60 stock | 画质（PSNR/SSIM） | 备注 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        quality = (f"{f(r['psnr'],2)} dB / {f(r['ssim'],3)}"
                   if r.get("psnr") is not None else "—")
        A(f"| {r['rank']} | {r['card']} | {r['type']} | **{f(r['step'],2)} s** | "
          f"{f(r['e2e'])} s | {f(r['vram'])} GB | {f(r['peak'])}/{f(r['avg'])} W | "
          f"{f(r['speedup'],2)}× | {quality} | {r['note']} |")
    A("")
    A("## 结论")
    A("")
    A("- **全场最快**：NVIDIA RTX PRO 4500 + SOL，9.8 s/步 / 115.3 s，比 B60 stock 快 **7.1×**。")
    A("- **画质优先**：NVIDIA RTX PRO 4500 + CK attention，16.6 s/步，PSNR 23.1 dB / SSIM 0.814。")
    A("- **Intel 最好**：B60 + Intel 镜像 SOL，37.0 s/步；比 stock B60 快 1.9×，但仍比 PRO 4500 + SOL 慢 3.8×。")
    A("- **AMD**：R9700 46.0 s/步，与 B70（41.2）和 B60 + Intel dense（46.4）同一档。")
    A("- **B60 stock 70.0 s/步不是硬件上限**：换 Intel 运行时后 dense 46.4、SOL 37.0。")
    A("")
    A("## 口径说明")
    A("")
    A("- B70 / R9700 原始为 1344×768，单步已乘 `1376/1344 = 1.0238`；端到端墙钟未归一化，且包含装载，不能与同机 NVIDIA 的纯计算时间直接相减。")
    A("- B60 stock 的功耗/显存有两套：主表用 20.2 GB / 181 W peak / 148 W avg；`perf.md` sysfs 为 21.4 GB / 139 W mean / 164 W peak。")
    A("- B60 + Intel dense/SOL 来自 Intel `llm-scaler-omni` 镜像，未采功耗；SOL 会改变采样轨迹。")
    A("- NVIDIA SOL/CK/Sage 的 PSNR/SSIM 是对 NVIDIA INT8 baseline（同 seed 1001）的逐帧比较；B60/B70/R9700 无同输入基线，画质列留空。")
    A("- W6A8 / FP8 在 32 GB 的 PRO 4500 上更慢（24.5 / 26.2 s/步），未列入本表；详见 `results/nvidia-optimizations.md`。")
    A("")

    path = os.path.join(ROOT, "results", "all-cards.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
