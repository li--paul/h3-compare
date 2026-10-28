# MiniMax-H3 全卡单段对比（Intel / AMD / NVIDIA）

统一条件：REF2VA / 124 帧 / 8 步 / CFG 1.0 / res_multistep+beta / seed 1001–1003 / INT8 ConvRot（除特别注明）。
按单步@1376×768 排序；相对加速以 Intel B60 stock 的 70.0 s/步为 1.00×。

| # | 卡 / 配置 | 类型 | 单步@1376 | 端到端 | 峰值显存 | 功耗（峰/均） | 相对 B60 stock | 画质（PSNR/SSIM） | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NVIDIA RTX PRO 4500 + SOL | NVIDIA 优化 | **9.80 s** | 115.3 s | 31.3 GB | 203.3/195.9 W | 7.14× | 18.01 dB / 0.713 | 同量化最快；tau=1.3；改变采样轨迹 |
| 2 | NVIDIA RTX PRO 4500 + SageAttention 2.2 | NVIDIA 优化 | **16.42 s** | 161.1 s | 31.1 GB | 203.6/198.8 W | 4.26× | 21.46 dB / 0.804 | 需源码编译 SageAttention 2.2.0 |
| 3 | NVIDIA RTX PRO 4500 + CK attention | NVIDIA 优化 | **16.57 s** | 162.3 s | 31.2 GB | 203.2/198.7 W | 4.22× | 23.05 dB / 0.814 | 画质优先；PSNR 最接近基线 |
| 4 | NVIDIA RTX PRO 4500（基线） | 基线 | **23.43 s** | 216.6 s | 31.1 GB | 203.4/198.0 W | 2.99× | — | PyTorch attention；32 GB aimdo 换入 |
| 5 | NVIDIA RTX 6000 Ada | 基线 | **24.04 s** | 223.3 s | 47.2 GB | 303.5/297.3 W | 2.91× | — | 48 GB 全驻留 |
| 6 | Intel Arc Pro B60 + SOL | XPU 优化 | **36.97 s** | 370.2 s | 21.4 GB | —/— W | 1.89× | — | Intel llm-scaler 镜像；稀疏注意力 |
| 7 | Intel Arc Pro B70 | 基线 | **41.25 s** | 478.3 s | 28.4 GB | 325.0/— W | 1.70× | — | 1344→1376 归一化；无 DynamicVRAM |
| 8 | AMD Radeon AI PRO R9700 | 基线 | **45.99 s** | 448.3 s | 28.1 GB | 293.0/— W | 1.52× | — | 1344→1376 归一化；开 DynamicVRAM |
| 9 | Intel Arc Pro B60 + Intel dense | XPU 优化 | **46.40 s** | 470.2 s | 21.4 GB | —/— W | 1.51× | — | Intel 镜像 dense；native INT8 + CUTE |
| 10 | Intel Arc Pro B60（stock） | 基线 | **70.00 s** | 653.0 s | 20.2 GB | 181.0/148.0 W | 1.00× | — | 交付包；INT8 仅存储、BF16 计算 |

## 结论

- **全场最快**：NVIDIA RTX PRO 4500 + SOL，9.8 s/步 / 115.3 s，比 B60 stock 快 **7.1×**。
- **画质优先**：NVIDIA RTX PRO 4500 + CK attention，16.6 s/步，PSNR 23.1 dB / SSIM 0.814。
- **Intel 最好**：B60 + Intel 镜像 SOL，37.0 s/步；比 stock B60 快 1.9×，但仍比 PRO 4500 + SOL 慢 3.8×。
- **AMD**：R9700 46.0 s/步，与 B70（41.2）和 B60 + Intel dense（46.4）同一档。
- **B60 stock 70.0 s/步不是硬件上限**：换 Intel 运行时后 dense 46.4、SOL 37.0。

## 口径说明

- B70 / R9700 原始为 1344×768，单步已乘 `1376/1344 = 1.0238`；端到端墙钟未归一化，且包含装载，不能与同机 NVIDIA 的纯计算时间直接相减。
- B60 stock 的功耗/显存有两套：主表用 20.2 GB / 181 W peak / 148 W avg；`perf.md` sysfs 为 21.4 GB / 139 W mean / 164 W peak。
- B60 + Intel dense/SOL 来自 Intel `llm-scaler-omni` 镜像，未采功耗；SOL 会改变采样轨迹。
- NVIDIA SOL/CK/Sage 的 PSNR/SSIM 是对 NVIDIA INT8 baseline（同 seed 1001）的逐帧比较；B60/B70/R9700 无同输入基线，画质列留空。
- W6A8 / FP8 在 32 GB 的 PRO 4500 上更慢（24.5 / 26.2 s/步），未列入本表；详见 `results/nvidia-optimizations.md`。

