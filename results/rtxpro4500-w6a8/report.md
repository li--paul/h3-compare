# MiniMax-H3 Ref2VA 1376×768 / 8 步 — NVIDIA RTX PRO 4500 Blackwell

由 `scripts/05_report.sh` 从 `timings.json` 自动生成；多卡对比见 `results/comparison.md`。

## 配置与运行参数

| 项 | 值 |
| --- | --- |
| 扩散主模型 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（20.97 GB） |
| 文本编码器 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（27.14 GB） |
| LoRA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| 分辨率 / 帧数 | 1376×768 / 124 帧（24 fps，约 5.17 s） |
| 采样 | res_multistep / beta / 8 步 / CFG 1.0 |
| 参考图 | `input/ref_1376x768.png`，`ref_image_size=match` |
| 硬件 | NVIDIA RTX PRO 4500 Blackwell，32623 MiB，功耗上限 200.00 W |
| ComfyUI | 0.40.0（XPU 侧为 0.39.0），comfy-kitchen 0.2.37 |
| 启动参数 | `run_tag=rtxpro4500-w6a8; port=8190; extra_args=; started=2026-10-28T09:55:46+08:00` |

## 分段耗时（秒）

| run | seed | 条件编码(TE+参考VAE) | 采样合计 | 单步中位数 | 单步范围 | 视频 VAE | 音频 VAE | 总计算 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold | 1001 | 5.3 | 195.7 | 24.5 | 21.5–27.2 | 20.4 | 0.2 | 227.4 |
| warm | 1002 | 4.8 | 195.6 | 24.5 | 21.5–27.1 | 20.4 | 0.2 | 225.9 |
| warm | 1003 | 4.8 | 195.7 | 24.5 | 21.5–27.1 | 20.5 | 0.2 | 226.3 |

## 与 Arc Pro B60 单卡对照

B60 数据来自 `h3单卡测试总结_20261022.md`（REF2VA / 1376×768 / 124 帧 / 8 步 / INT8）。

| 指标 | B60 | 本机中位数 | 本机/XPU（耗时比） |
| --- | --- | --- | --- |
| 单步耗时 | 70 s | 24.5 s | 0.35× |
| 8 步采样 | 563 s | 195.7 s | 0.35× |
| 条件编码 | 16.4 s | 4.8 s | — |
| VAE 解码 | 73.6 s | 20.6 s | 0.28× |
| 单段总计 | 653 s | 226.3 s | 0.35× |

- 按单步计，本机比 B60 快 **2.9 倍**。
- 按 8 步采样计，快 **2.9 倍**。
- 按单段总计计，快 **2.9 倍**。

## 实测运行状态（1 Hz 采样）

| 指标 | 值 |
| --- | --- |
| 峰值显存 | 31837 MiB / 32623 MiB（98%） |
| 峰值功耗 | 204 W / 200.00 W |
| 平均功耗 | 197 W |
| GPU 利用率（>50% 样本） | 均值 99%（669/680 样本） |
| SM 时钟范围 | 2085–2805 MHz |
| 原始数据 | `logs/rtxpro4500-w6a8/nvidia-smi_20261028_091715.csv` |

## 输出校验

| 文件 | 视频 | 帧数 / fps | 音频 | 时长 | 大小 |
| --- | --- | --- | --- | --- | --- |
| `output/rtxpro4500-w6a8/bench/h3_ref2va_1376x768_00001_.mp4` | h264 1376×768 | 124 / 24/1 | aac 32000 Hz 2ch | 5.17 s | 2.5 MB |
| `output/rtxpro4500-w6a8/bench/h3_ref2va_1376x768_00002_.mp4` | h264 1376×768 | 124 / 24/1 | aac 32000 Hz 2ch | 5.17 s | 3.5 MB |
| `output/rtxpro4500-w6a8/bench/h3_ref2va_1376x768_00003_.mp4` | h264 1376×768 | 124 / 24/1 | aac 32000 Hz 2ch | 5.17 s | 3.8 MB |

## 口径与限制

- **权重逐字节一致**：与 XPU 交付包同名的 INT8 ConvRot 文件，字节数与 `results/sha256.txt` 已核对。
- **算子路径不同**：CUDA 上 `int8_tensorwise` 为 native INT8 GEMM；XPU 上该算子被禁用，权重反量化成 BF16。属于“同权重、不同 kernel”。
- **显存策略不同**：B60 为 24 GB + 流式换入；本机按实际显存容量驻留/换出，启动参数见上表。
- **输入差异**：B60 用《神笔马良》第 1 段提示词，本机用《草船借箭》提示词；token 数同量级，但非逐字节同输入。
- **版本差**：ComfyUI 0.40.0 vs 0.39.0；torch 2.14.1 基础版本一致；comfy-kitchen 均为 0.2.37。
- **缓存规避**：每次 run 更换 seed（1001/1002/1003）并把提示词 `[Shot 1]` 改为 `[Shot N]`。

