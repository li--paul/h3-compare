# MiniMax-H3 单卡出片基准（B60 / Ada / PRO 4500 + B70 / R9700 外部参考）

按 XPU 交付包（`MiniMax-H3_环境与模型版本清单.md`）的 Ref2VA 默认配置，在不同卡上
跑同一组 INT8 ConvRot 权重，比较分段耗时、显存和功耗。

当前结论见 `results/comparison.md`：

| 卡 | 单步 | 单段 | 峰值显存 | 平均功耗 | 每步能耗 | 段能耗 |
|---|---|---|---|---|---|---|
| Arc Pro B60 24 GB | 70.0 s | 653 s | 20.2 GB | 148 W | 12.7 kJ | 26.8 Wh |
| RTX 6000 Ada 48 GB | 24.0 s | 223 s | 47.2 GB | 297 W | 7.1 kJ | 18.4 Wh |
| RTX PRO 4500 32 GB | 23.4 s | 217 s | 31.1 GB | 198 W | 4.6 kJ | 11.9 Wh |

PRO 4500 与 Ada 基本持平（3% 以内），但功耗约为 2/3。B60 的 70 s/步是 **stock ComfyUI** 的结果；
换 Intel 运行时后同一张 B60 可到 46.4 s/步（dense）或 37.0 s/步（SOL）。

### XPU 运行时分层（同一张 B60）

| 路径 | 单步 | 端到端 | 说明 |
|---|---:|---:|---|
| ComfyUI 0.39 / 0.42 stock | 70.1 s | 652 s | 交付包基准；INT8 仅存储，BF16 计算 |
| Intel `llm-scaler-omni` 镜像 dense | 46.4 s | 470 s | native INT8 + CUTE attention，1.51× |
| 同上 + SOL（start=0.0） | 37.0 s | 370 s | 稀疏注意力，1.90×；会改变采样轨迹 |

来源：<https://github.com/li--paul/h3-comfyui-xpu> 的 `perf.md`；本地存档在
`results/sources/h3-comfyui-xpu-perf.md`。

外部参考（服务器 A，1344×768，单步已归一化到 1376×768）：

| 卡 | 单步@1376 | 8 步去噪 | 墙钟 | 每步能耗 |
|---|---|---|---|---|
| Arc Pro B70 | 41.2 s | 322 s | 478 s | 13.4 kJ |
| Radeon R9700 默认 | 46.0 s | 359 s | 448 s | 13.5 kJ |
| R9700 nodyn | 46.0 s | 360 s | 690 s | 13.2 kJ |

**墙钟不可跨主机比较**：服务器 A 是 61 GB 内存 + PCIe Gen4，且 B70 没有 XPU 版 DynamicVRAM。
去噪算力排序（含运行时分层）：PRO 4500 ≈ Ada > B60+SOL > B70 > R9700 ≈ B60+Intel dense > B60 stock。
每步能耗则是 PRO 4500 < Ada < B60 < B70 ≈ R9700。

### NVIDIA 运行时优化（RTX PRO 4500，同 INT8 权重）

| 配置 | 单步 | 端到端 | 相对基线 | PSNR / SSIM | 说明 |
|---|---:|---:|---:|---|---|
| INT8 baseline | 23.4 s | 217 s | 1.00× | — | PyTorch attention |
| **INT8 + SOL tau=1.3** | **9.8 s** | **115 s** | **2.39×** | 18.0 dB / 0.713 | 推荐；同量化，最快 |
| INT8 + SOL tau=2.0 | 9.1 s | 105 s | 2.58× | 14.3 dB / 0.600 | 太快导致构图偏移，不推荐 |
| INT8 + CK attention | 16.6 s | 162 s | 1.41× | 23.1 dB / 0.814 | 画质优先 |
| INT8 + SageAttention 2.2 | 16.4 s | 161 s | 1.43× | 21.5 dB / 0.804 | 需源码编译 2.2.0 |
| W6A8（官方） | 24.5 s | 226 s | 0.96× | 20.4 dB / 0.789 | 32 GB 上更慢 |
| FP8 scaled（官方） | 26.2 s | 240 s | 0.89× | 20.8 dB / 0.758 | 32 GB 上更慢 |

完整表格与启动参数见 `results/nvidia-optimizations.md`。

- SOL 用 `BlockSparseAttention` 节点，`selection=sol-attn`、`tau=1.3`、`start_percent=0.0`、`sink_conditioning=exact_kv_and_rows`。
- SageAttention 2.2.0 不在 PyPI；需从 `thu-ml/SageAttention@v2.2.0` 源码编译（C++20；CUDA 13.1 + 12.8 的 cuBLAS/cuSPARSE 头）。
- `--fast cublas_ops autotune` 只有 1.01×，且同 seed 输出会变化，不建议作为默认。

## 一键运行

每张卡的数据按 `RUN_TAG` 隔离：

```bash
cd /home/acm/work/nv_h3
RUN_TAG=rtxpro4500 bash scripts/run_all.sh   # 下载 -> 准备 -> 启动 -> 冷+2热 -> 报告 -> 对比 -> 停止
```

单独阶段：

```bash
RUN_TAG=rtxpro4500 bash scripts/01_download.sh  # 下载 INT8 ConvRot 权重与 8 步 LoRA（约 56 GB）
RUN_TAG=rtxpro4500 bash scripts/02_prepare.sh   # 抽参考帧、写 config、记录环境
RUN_TAG=rtxpro4500 bash scripts/03_serve.sh     # 启动 ComfyUI (127.0.0.1:8190)
RUN_TAG=rtxpro4500 bash scripts/04_bench.sh     # 冷启动 + 2 次热跑
RUN_TAG=rtxpro4500 bash scripts/05_report.sh    # 生成 results/<tag>/report.md
bash scripts/06_compare.sh                     # 生成 results/comparison.md / .csv
bash scripts/stop.sh
```

`RUN_TAG` 默认是 `rtxpro4500`。已有归档：`rtx6000ada`。

所有执行过的命令（含时间戳和退出码）追加在 `logs/commands.log`；
每次运行的环境快照和卡信息在 `logs/<tag>/{env_*,card.txt,run_flags.txt}`。

## 固定参数

| 参数 | 值 |
| --- | --- |
| 分辨率 / 帧数 | 1376×768 / 124 |
| 采样 / 调度 | res_multistep / beta |
| 步数 / CFG | 8 / 1.0 |
| 种子 | 1001、1002、1003（每次不同，绕开 ComfyUI 节点缓存） |
| LoRA | ref2v turbo 8step v1.0 768p |
| 权重 | `minimax_h3_ref2va_pruned_int8_convrot` + `qwen3vl_32b_..._int8_convrot` |

## 目录

```
models/                       # 与 XPU 同名的 INT8 ConvRot 权重（sha256 在 results/）
input/                        # 提示词与 1376×768 参考图
workflows/                    # ComfyUI API 工作流
results/<tag>/                # timings.json / timings.csv / report.md
results/comparison.md         # 多卡总表
results/sources/              # B60 实测总结、B70/R9700 对比报告等原始数据
output/<tag>/bench/*.mp4      # 1376×768 / 124 帧 / 24 fps / AAC
logs/<tag>/                   # comfy 日志、nvidia-smi 采样、逐次 websocket 事件
logs/commands.log             # 全部命令留痕
```

## 注意

- CUDA 上 INT8 权重走 native INT8 GEMM；XPU 上 `supports_int8_compute()` 为 False，
  权重会反量化成 BF16 再计算。权重相同、kernel 不同，报告里单独标注。
- 48 GB 的 Ada 可全驻留；32 GB 的 PRO 4500 由 comfy-aimdo 自动换入换出（峰值 31.1 GB），
  本轮没有 OOM、没有降级。
- 换卡后重新跑时，用新的 `RUN_TAG`，不要覆盖已有归档。
- 外部参考表只用于量级对照：不同主机、不同分辨率（已按 1.0238 归一化）、不同软件栈，
  不要把 B70/R9700 的墙钟和本机数字直接相减。
