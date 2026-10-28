# MiniMax-H3 Ref2VA 单卡性能对比（同机 + 外部参考 + XPU 运行时分层）

统一条件：REF2VA / 124 帧 / 8 步 / CFG 1.0 / res_multistep+beta / batch 1 / INT8 ConvRot。
同机三行使用同一份输入、同一工作流、同一组权重（sha256 见 `results/sha256.txt`）。

## 同机基准（1376×768）

| 卡 | 显存 | 功耗上限 | 精度 / kernel | 单步@1376 | 8 步采样 | 条件编码 | VAE | 单段总计 | 峰值显存 | 功耗（峰/均） | 每步能耗 | 段能耗 | 段/小时 | s/Gpx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intel Arc Pro B60 | 24 GB | — | INT8 ConvRot / XPU 反量化→BF16 | 70.0 s | 563.0 s | 16.4 s | 73.6 s | **653.0 s** | 20.2 GB | 181.0/148.0 W | 12.7 kJ | 26.8 Wh | 5.51 | 4983.24 |
| NVIDIA RTX 6000 Ada Generation | 48.0 GB | 300.00 W | INT8 ConvRot / CUDA INT8 GEMM | 24.0 s | 189.3 s | 5.1 s | 22.8 s | **223.3 s** | 47.2 GB | 303.5/297.3 W | 7.1 kJ | 18.4 Wh | 16.12 | 1704.31 |
| NVIDIA RTX PRO 4500 Blackwell | 31.9 GB | 200.00 W | INT8 ConvRot / CUDA INT8 GEMM | 23.4 s | 185.3 s | 4.9 s | 20.1 s | **216.6 s** | 31.1 GB | 203.4/198.0 W | 4.6 kJ | 11.9 Wh | 16.62 | 1652.72 |

> 每步能耗 = 单步 × 忙碌功耗；B60 用峰值 181 W，NVIDIA 用忙碌均值。
> 段能耗 = 平均功耗 × 单段，含装载/解码，仅同机三行可比。

## 外部参考（服务器 A，1344×768）

来源：`results/sources/report-r9700-vs-b70.md`（2026-10-23）。

| 跑次 | 原始单步 | 单步@1376 | 8 步去噪 | 墙钟 | 显存峰值 | 忙碌功耗 | 每步能耗@1376 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intel Arc Pro B70 | 40.29 s | 41.2 s | 322.3 s | 478.3 s | 28.4 GB | 325.0 W | 13.4 kJ | 无 DynamicVRAM；仅 1 个有效跑次 |
| AMD Radeon AI PRO R9700（默认） | 44.92 s | 46.0 s | 359.4 s | 448.3 s | 28.1 GB | 293.0 W | 13.5 kJ | hip + DynamicVRAM（pinned 54.6 GB） |
| R9700 nodyn（关 DynamicVRAM） | 44.97 s | 46.0 s | 359.8 s | 690.1 s | 28.7 GB | 286.0 W | 13.2 kJ | 与 B70 同为旧式装载，墙钟被读盘拖累 |
| R9700 ckt（triton 优先） | 46.14 s | 47.2 s | 369.1 s | 509.5 s | 30.2 GB | 289.0 W | 13.7 kJ | COMFY_CK_PRIORITY=triton,hip,eager；整段反而更慢 |

- 归一化系数 `1376×768 / 1344×768 = 1.0238`；外部表的单步和每步能耗都乘了这个系数。
- `R9700 triton` 与默认实际同走 hip 后端（`--enable-triton-backend` 只加入候选、不改优先级），
  单步 44.95 s、墙钟 443.9 s，不单独成行。
- **墙钟不可跨主机比较**：服务器 A 是 61 GB 内存 + PCIe Gen4，本机 183 GB + Gen5；
  B70 还没有 XPU 版 DynamicVRAM。B70 478 s 与 R9700 默认 448 s 的差距主要来自装载方式，
  不是去噪算力；关掉 DynamicVRAM 后 R9700 变成 690 s（比 B70 慢 212 s）。
- 数据质量：B70 只有 1 个有效跑次；R9700 nodyn 成片亮度异常（110.1 vs 117.0）未查；
  R9700 默认跑次的利用率列是中途补采的。

## XPU 运行时分层（同一张 B60，1376×768）

来源：`https://github.com/li--paul/h3-comfyui-xpu`（`perf.md`，2026-10-22/27）。

| 路径 | 精度 / kernel | 单步 | 8 步采样 | 端到端 | 峰值显存 | 相对 stock | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ComfyUI 0.39 交付包 | INT8 ConvRot / stock BF16 | 70.10 s | 560.8 s | 652.0 s | 21.4 GB | 1.00× | 现有 B60 基准 |
| ComfyUI 0.42（h3-comfyui-xpu） | INT8 ConvRot / stock BF16 | 70.10 s | 560.8 s | 652.0 s | 21.4 GB | 1.00× | 与 0.39 逐像素一致 |
| B60 + Intel 镜像 dense | native INT8 + CUTE | 46.40 s | 371.2 s | 470.2 s | 21.4 GB | 1.51× | 1.51× vs stock（仓库自述 1.39×） |
| B60 + SOL（start=0.0） | native INT8 + SOL | 36.97 s | 295.8 s | 370.2 s | 21.4 GB | 1.90× | 1.27× vs dense；改变采样轨迹 |

- stock ComfyUI 的 INT8 只是存储格式，matmul 走 BF16；Intel 镜像才提供 native `int8_linear` + CUTE attention。
- 仓库自述“1.39×”，但按 70.1 / 46.4 计算为 1.51×；这里以原始单步为准。
- SOL 会改变采样轨迹（相对 dense 的 PSNR 18–26 dB），画面结构一致但细节不同，**不是同配置**。
- Intel 镜像 dense / SOL 未测功耗，因此不进入“每步能耗”排序。

### 其他运行时（不同卡数/图，不并入单卡排序）

| 路径 | 卡数 | 端到端 | 显存/卡 |
| --- | --- | --- | --- |
| vLLM-Omni FP8 + LoRA | 2 | 338–349 s | 14.7 GiB/rank |
| vLLM-Omni BF16 + LoRA | 2 | 375 s | 18.7 GiB/rank |
| diffusers BF16 layer-streamed | 1 | 920 s | 17.6 GiB |

- vLLM-Omni 是 2 卡 TP2 + DLO；diffusers 是单卡 layer-streamed。两者都不是单卡 ComfyUI 的同口径对比。

## 跨报告结论（归一化后）

### 去噪算力

| 排名 | 卡 | 单步@1376 | 耗时/B70（越小越快） | 来源 |
| --- | --- | --- | --- | --- |
| 1 | NVIDIA RTX PRO 4500 Blackwell | 23.4 s | 0.57× | 同机 |
| 2 | NVIDIA RTX 6000 Ada Generation | 24.0 s | 0.58× | 同机 |
| 3 | B60 + SOL（start=0.0） | 37.0 s | 0.90× | XPU 运行时 |
| 4 | Intel Arc Pro B70 | 41.2 s | 1.00× | 服务器 A |
| 5 | AMD Radeon AI PRO R9700（默认） | 46.0 s | 1.11× | 服务器 A |
| 6 | R9700 nodyn（关 DynamicVRAM） | 46.0 s | 1.12× | 服务器 A |
| 7 | B60 + Intel 镜像 dense | 46.4 s | 1.12× | XPU 运行时 |
| 8 | R9700 ckt（triton 优先） | 47.2 s | 1.15× | 服务器 A |
| 9 | Intel Arc Pro B60 | 70.0 s | 1.70× | 同机 |

- PRO 4500 与 Ada 基本持平（3% 以内）；两者比 B70 快约 1.76× / 1.72×。
- **B60 的 stock 70.1 s/步不是硬件上限**：Intel 镜像 dense 46.4 s（1.51×）、
  SOL 36.97 s（1.90×）。换运行时后，B60 与 PRO 4500 的差距从 3.0× 缩到 1.6×（SOL）/ 2.0×（dense）。
- B70 的 40.29 s/步仍是 eager 后端；若换 Intel 镜像可能继续下降（未测，不写入表）。
- B70 比 R9700 默认快约 11%；R9700 nodyn 与默认去噪相同，慢在装载。

### 每步能耗

| 卡 | 每步能耗 | 能耗/B70（越小越省） |
| --- | --- | --- |
| NVIDIA RTX PRO 4500 Blackwell | 4.6 kJ | 0.35× |
| NVIDIA RTX 6000 Ada Generation | 7.1 kJ | 0.53× |
| Intel Arc Pro B60 | 12.7 kJ | 0.95× |
| R9700 nodyn（关 DynamicVRAM） | 13.2 kJ | 0.98× |
| Intel Arc Pro B70 | 13.4 kJ | 1.00× |
| AMD Radeon AI PRO R9700（默认） | 13.5 kJ | 1.01× |
| R9700 ckt（triton 优先） | 13.7 kJ | 1.02× |

- PRO 4500 每步能耗约为 B70 的 0.35×，Ada 约 0.53×，B60 约 0.95×。
- B70 与 R9700 默认几乎相同（13.4 vs 13.5 kJ/步）；R9700 ckt 略差。

## 同机相对 B60 的加速

| 卡 | 单步 | 8 步采样 | 单段总计 | 段能耗（B60/本机） |
| --- | --- | --- | --- | --- |
| NVIDIA RTX 6000 Ada Generation | 2.91× | 2.97× | 2.92× | 1.46× |
| NVIDIA RTX PRO 4500 Blackwell | 2.99× | 3.04× | 3.02× | 2.25× |

## RTX PRO 4500 vs RTX 6000 Ada

- 单步：PRO 4500 快 **1.03×**（23.4 s vs 24.0 s）
- 单段：PRO 4500 快 **1.03×**（216.6 s vs 223.3 s）
- 峰值显存：Ada 47.2 GB vs PRO 4500 31.1 GB
- 段能耗：Ada 18.4 Wh vs PRO 4500 11.9 Wh （PRO 4500 省 1.55×）

两者单步差距在 3% 以内，已在轮次波动范围内，应视为**基本持平**；
PRO 4500 用 2/3 的功耗和 2/3 的显存达到同样速度。

## 口径说明

- **B60 数据**：`results/sources/h3单卡测试总结_20261022.md`，.118 机器单卡 REF2VA；
  提示词为《神笔马良》第 1 段，与 NVIDIA 两轮的《草船借箭》不是逐字节同输入。
- **B60 功耗/显存两套数据**：主表用 20.2 GB / 181 W peak / 148 W avg；
  `h3-comfyui-xpu/perf.md` 用 sysfs 测得 21.4 GB / 139 W mean / 164 W peak。
  采样方法不同（原采样器 vs xe sysfs），不要混算或直接替换。
- **XPU 运行时分层**：`results/sources/h3-comfyui-xpu-perf.md`；Intel 镜像是容器运行时，
  native INT8 + CUTE/SOL，不等同于交付包或本机 CUDA 路径。SOL 改变采样轨迹，不是同配置。
- **同权重、不同 kernel**：CUDA 上 `int8_tensorwise` 是 native INT8 GEMM；Intel XPU 上该算子被禁用，
  权重反量化后以 BF16 计算；R9700 走 ROCm hip。加速比不应全部归因于硬件。
- **Blackwell 原生算子**：PRO 4500 上 `int8_tensorwise`、`convrot_w4a4`、`nvfp4`、`mxfp8` 均为 native；
  Ada 上 `nvfp4` / `mxfp8` 为 emulated。本轮两边都只用 INT8，未使用 FP4。
- **显存策略不同**：B60 24 GB 流式换入；NVIDIA 两轮都启用 comfy-aimdo 0.5.6 动态显存——
  Ada 48 GB 可全驻留（峰值 48.3 GB），PRO 4500 32 GB 在 31.1 GB 峰值下自动换入换出，但未 OOM、未降级。
  实际启动参数写在 `logs/<tag>/run_flags.txt`（本轮为空，即默认 `--bf16-vae`）。
- **软件版本**：B60 为 ComfyUI 0.39.0 + JZL 节点；NVIDIA 为 0.40.0 + 核心节点；
  comfy-kitchen 均为 0.2.37。服务器 A 报告为 comfy-kitchen 0.2.39，B70 走 XPU eager，R9700 走 hip。
- **输出一致性**：同 seed 1001 下 Ada 与 PRO 4500 的画面结构一致、细节不同
  （PSNR≈19.7 dB，SSIM≈0.79），属于不同 INT8 kernel 的数值分歧；不要用逐像素一致做验收。

