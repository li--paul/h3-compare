# MiniMax-H3 单卡出片基准（B60 / Ada / PRO 4500 + B70 / R9700 外部参考）

按 XPU 交付包（`MiniMax-H3_环境与模型版本清单.md`）的 Ref2VA 默认配置，在不同卡上
跑同一组 INT8 ConvRot 权重，比较分段耗时、显存和功耗。

当前结论见 `results/comparison.md`：

| 卡 | 单步 | 单段 | 峰值显存 | 平均功耗 | 每步能耗 | 段能耗 |
|---|---|---|---|---|---|---|
| Arc Pro B60 24 GB | 70.0 s | 653 s | 20.2 GB | 148 W | 12.7 kJ | 26.8 Wh |
| RTX 6000 Ada 48 GB | 24.0 s | 223 s | 47.2 GB | 297 W | 7.1 kJ | 18.4 Wh |
| RTX PRO 4500 32 GB | 23.4 s | 217 s | 31.1 GB | 198 W | 4.6 kJ | 11.9 Wh |

PRO 4500 与 Ada 基本持平（3% 以内），但功耗约为 2/3；两者都比 B60 快约 3 倍。

外部参考（服务器 A，1344×768，单步已归一化到 1376×768）：

| 卡 | 单步@1376 | 8 步去噪 | 墙钟 | 每步能耗 |
|---|---|---|---|---|
| Arc Pro B70 | 41.2 s | 322 s | 478 s | 13.4 kJ |
| Radeon R9700 默认 | 46.0 s | 359 s | 448 s | 13.5 kJ |
| R9700 nodyn | 46.0 s | 360 s | 690 s | 13.2 kJ |

**墙钟不可跨主机比较**：服务器 A 是 61 GB 内存 + PCIe Gen4，且 B70 没有 XPU 版 DynamicVRAM。
去噪算力排序为 PRO 4500 ≈ Ada > B70 > R9700 > B60；每步能耗则是 PRO 4500 < Ada < B60 < B70 ≈ R9700。

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
