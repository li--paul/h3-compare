# MiniMax-H3 ComfyUI 单卡对比：Intel Arc Pro B70 与 AMD Radeon AI PRO R9700

2026-10-23，服务器 A `192.168.0.14`（lm-linux）。

## 结论

1. **算力**：B70 每步去噪 40.3 秒，R9700 44.9 秒，B70 快 11%。B70 功耗也高 11%（325 W 对 293 W），所以每步能耗基本相同（约 13.1 kJ 对 13.2 kJ）。
2. **默认设置下的墙钟**：R9700 448 秒，B70 478 秒，R9700 少 30 秒。这 30 秒全部来自 DynamicVRAM：ComfyUI 在 ROCm 上开 DynamicVRAM，在 xpu 上不开。
3. **关掉 DynamicVRAM，两卡装载方式相同时**：R9700 690 秒，比 B70 慢 212 秒。其中去噪只占 37 秒，其余约 175 秒是装载开销：主机内存涨满（60.9/61 GB），页缓存被挤掉，权重要重复读盘。
4. **int8 后端**：R9700 默认走 hip，这是最快的选择。只加 `--enable-triton-backend` 不起作用。把 triton 设为最高优先级后，每步反而慢 2.7%。
5. **成片**：1344×768，124 帧，带 aac 音轨，没有黑帧。R9700 nodyn（关 DynamicVRAM）的平均亮度是 110.1，其余 R9700 跑次都是 117.0，种子相同。原因未查。

## 跑次代号

| 代号 | 显卡 | 含义 | 和"R9700 默认"相比只改了什么 |
|---|---|---|---|
| **B70** | Intel B70 | Intel 卡基准跑次，记录目录名 `comfyui-int8-lora-ra16m`（ra16m 表示当时磁盘预读设为 16 MB） | 不同的卡、驱动和软件栈 |
| **R9700 默认** | AMD R9700 | ComfyUI 原样配置：int8 用 AMD 的 hip 核函数，打开 DynamicVRAM（动态显存管理） | — |
| **R9700 triton** | AMD R9700 | 加了参数 `--enable-triton-backend`，想改用 triton 编写的 int8 核函数 | 实际仍走 hip，参数没有生效 |
| **R9700 nodyn** | AMD R9700 | no dynamic VRAM：关掉 DynamicVRAM，让权重装载方式和 B70 一样 | 加 `--disable-dynamic-vram` |
| **R9700 ckt** | AMD R9700 | ck = comfy-kitchen（ComfyUI 的量化算子库），t = triton：把 triton 强制设为第一优先 | 加 `--enable-triton-backend` 和环境变量 `COMFY_CK_PRIORITY=triton,hip,eager` |

名词说明：
- **DynamicVRAM**：ComfyUI 的动态显存管理，由 comfy-aimdo 实现。权重直接从系统缓存搬到显存，不在进程内存里另存一份。ComfyUI 只在 NVIDIA 和 AMD 卡上启用，Intel 卡上用不了。
- **hip / triton / eager**：同一个 int8 运算的三种实现。hip 是 AMD 预编译的核函数；triton 在运行时编译；eager 由普通 PyTorch 算子拼成，速度最慢，B70 用的就是 eager。
- **int8 ConvRot**：模型权重的 int8 量化格式。量化前先对每 256 个通道做一次旋转，把极端值打散，减小精度损失。

## 测试条件

两卡相同的部分（见 `~/h3/h3-b70-bench/spec.md`）：
- 输入：shot01，ref2va，参考图 1 张，H3_PROMPT 2386 字。
- 生成参数：1344×768，124 帧，24 fps，种子 1001，8 步，采样器 res_multistep，调度 beta，cfg 1.0。
- 模型：
  - DiT：`minimax_h3_ref2va_pruned_int8_convrot.safetensors`，21 GB。
  - 文本编码器：`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`，27 GB。
  - 视频 VAE fp16，音频 VAE fp32。
  - LoRA：ref2v Turbo 8 步 768p，强度 1.0。
- 调用链：`common/run_one.py` → `comfyui/run.py` → `~/comfyui-tools/run_shots.py`。
- 平台：PCIe Gen4 ×16（插槽侧），主机内存 61 GB，NVMe `nvme0n1`。

| 项 | B70 | R9700 |
|---|---|---|
| 显卡 | Arc Pro B70，32 GB | Radeon AI PRO R9700（Navi 48，gfx1201），32 GB |
| 驱动 | 内核自带 `xe` | `amdgpu-dkms` 7.1.9（31.60） |
| 运行时 | oneAPI 2026.1 | 系统 ROCm 10.1，torch wheel 自带 ROCm 7.14 |
| torch | `2.14.1+xpu`，`~/comfyui/.venv` | `2.14.1+rocm7.14`，`~/comfyui/.venv-rocm` |
| comfy-kitchen 0.2.39 后端 | eager | hip |
| DynamicVRAM | 关（xpu 不支持） | 开，pinned 上限 54.6 GB |
| `read_ahead_kb` | 16384 | 128。预热已改为并行 pread，不受这个值影响 |

## 时间拆分（A+B+C+D+E+F+G+H = 墙钟，单位秒）

| 阶段 | B70 | R9700 默认 | R9700 triton | R9700 nodyn | R9700 ckt |
|---|---|---|---|---|---|
| A 脚本准备（选卡） | 1.8 | 1.1 | 1.3 | 0.6 | 0.6 |
| B 预热页缓存 | 14.0 | 4.0 | 6.0 | 7.0 | 30.0 |
| C 启动 ComfyUI | 10.0 | 12.0 | 6.0 | 6.0 | 8.0 |
| D 装文本编码器＋编码＋卸载 | 55.1 | 33.0 | 36.8 | 56.5 | 36.5 |
| E 装 DiT | 11.1 | 1.2 | 1.0 | 60.5 | 21.6 |
| F 去噪 8 步 | **322.3**（40.29×8） | 359.4（44.92×8） | 359.6（44.95×8） | 359.8（44.97×8） | 369.1（46.14×8） |
| G 卸 DiT＋VAE 解码＋封装 | 62.5 | 36.4 | 32.6 | 198.2 | 42.8 |
| H 收尾（报告、停采样器） | 1.5 | 1.2 | 0.7 | 1.5 | 0.9 |
| **合计＝墙钟** | **478.3** | **448.3** | **443.9** | **690.1** | **509.5** |

切分方法（脚本 `common/stage_sum.py`）：
- A、B、C、H 取自 `cmd.log` 的秒级时间戳。
- F = 8 × 进度条的每步秒数。去噪结束点取显存 ≥ 22 GiB 的最长连续区间的末端，F 的起点由结束点往前推 8 步得到。
- D 的结束点：开始调度后，显存先升到 10 GiB 以上，再第一次跌到峰值一半以下的那个采样点。
- E = D 结束到 F 开始，G = F 结束到 shot01 完成。
- D、E、F、G 之间的边界依赖每 5 秒一次的显存采样，单个边界误差约 ±5 秒，各项合计仍等于墙钟。
- DynamicVRAM 开启时 DiT 是按需装载的，装载时间多半算进了第 1 步，所以 E 只有约 1 秒，F 的每步时间略偏高。

## 资源与成片

| 项 | B70 | R9700 默认 | R9700 triton | R9700 nodyn | R9700 ckt |
|---|---|---|---|---|---|
| 设置 | — | hip，DynamicVRAM 开 | 加 `--enable-triton-backend` | 加 `--disable-dynamic-vram` | triton 设为最高优先级 |
| 进程 RSS 峰值 | 59.2 GB | 9.0 GB | 9.0 GB | 59.2 GB | 9.1 GB |
| major page faults | 334,408 | 2,325 | — | 111,297 | — |
| 读盘量 | 97.6 GB | 0.56 GB | 3.7 GB | 29.7 GB | 55.4 GB（多为预热） |
| 主机内存峰值 | 38.8 GB | 7.9 GB | 9.3 GB | 60.9 GB | 9.2 GB |
| 显存峰值 | 28.4 GB | 28.1 GB | 28.1 GB | 28.7 GB | 30.2 GB |
| 忙碌时功耗 / 频率 | 325 W / 2648 MHz | 293 W / 2580 MHz | 292 W / 2578 MHz | 286 W / 2597 MHz | 289 W / 2630 MHz |
| 进程 CPU | 115% | 234% | 234% | — | — |
| 成片亮度（均值 / 最暗帧） | 未测 | 117.0 / 113.8 | 117.0 / 113.8 | 110.1 / 107.0 | 117.1 / 113.8 |

- 读盘量 = File system inputs × 512 B。
- 忙碌时 = GPU 利用率超过 50% 的采样点，每 5 秒采一次。

## 分析

### 去噪算力

单算子测试：int8 线性层，M=16384，K=N=5120，脚本 `common/smoke_int8_rocm.py`。

| 实现 | 耗时 | 吞吐 | 相对误差 |
|---|---|---|---|
| bf16 matmul | 7.30 ms | 117.6 TFLOPS | 基准 |
| int8 hip | 5.10 ms | 168 TOPS | 1.27% |
| int8 triton | 4.19 ms | 205 TOPS | 1.30% |
| int8 eager | 14.3 ms | 60 TOPS | 1.30% |

换成 DiT 的真实形状再测一次：M=32768，开 convrot，脚本 `common/ck_backend_bench.py`。

| K×N | hip | triton |
|---|---|---|
| 5120×5120 | 11.44 ms | 9.90 ms |
| 5120×13824 | 27.73 ms | 23.10 ms |
| 13824×5120 | 29.09 ms | 27.32 ms |

### int8 走哪个后端

- 在 AMD 设备上，`comfy_kitchen/__init__.py:59` 把后端优先级设为 `["hip", "cuda", "triton", "eager"]`。实测 `get_capable_backend("int8_linear")` 返回 hip（脚本 `common/ck_which_backend.py`）。`--enable-triton-backend` 只把 triton 加入可选列表，不改优先级，所以 R9700 triton 和 R9700 默认走的是同一组核函数。
- R9700 ckt 用 `COMFY_CK_PRIORITY=triton,hip,eager` 把 triton 排到最前，日志确认已生效。单算子测试里 triton 快 6% 到 17%，整段却慢 2.7%。推测原因：rope、adaln、rms 等非 int8 算子也被切到了 triton，而这些算子在 triton 上比 hip 慢。这一点未验证。

### 去噪以外的时间

- 去噪以外的差距主要在 D、E、G 三段。B70 是 55+11+63=129 秒，R9700 默认是 33+1+36=70 秒，R9700 nodyn（关 DynamicVRAM）是 57+61+198=316 秒。
- `~/comfyui/main.py:275` 的 `dynamic_vram_supported()` 只在两种情况下返回 True：NVIDIA，或 ROCm ≥ 7.14 的 AMD。
- **不开 DynamicVRAM**（B70 和 R9700 nodyn 都属于这种情况）：
  - 走旧的 ModelPatcher，权重整份读进进程内存，RSS 到 59 GB。
  - 55.9 GB 的权重页缓存因此被挤掉，权重要重复读盘。
  - R9700 nodyn 比 B70 多花约 190 秒，主机内存也被占满。推测 ROCm 的 pinned 分配比 xpu 更占内存。这一点未验证。
- **开 DynamicVRAM**（comfy-aimdo）：权重从页缓存经 pinned 内存直接传到显存，RSS 9 GB，几乎不读盘。

## 已知问题和不可比项

1. **预热时间不能横比。** B70 跑次用的是旧版 `warm_weights`：每个文件一个线程顺序读，速度受 `read_ahead_kb` 影响。R9700 用的是新版：16 线程、64 MiB 一块的 pread。各跑次开始时页缓存状态也不同。预热不计入 Prompt executed。
2. **R9700 默认跑次的 `gpu.csv` 利用率一列全是 0。** 第一版 AMD 采样器读 DRM fdinfo，但 ROCm 的计算队列走 KFD，不会出现在 fdinfo 里。已改为读 sysfs `gpu_busy_percent`。这次跑次的利用率另存在 `gpu_busy.csv`，是中途补开的，一直采到下一个跑次。表中这一跑次的功耗和频率只取它自己时间段内的样本。
3. **R9700 第一次启动失败**，报 `TypeError: %d format`，原因是启动命令拼接的 bug。已修好，失败那次的跑次已删除。
4. **B70 只有一个有效跑次。** 另一次 678 秒的跑次按用户要求删除了。
5. **R9700 nodyn 的成片亮度和其他跑次对不上**（110.1 对 117.0），原因未查。

## 改动清单（服务器 A）

| 位置 | 改动 |
|---|---|
| `~/comfyui-tools/run_shots.py` | `warm_weights` 改为 16 线程、64 MiB 一块的并行 pread。新增 `JZL_GPU_BACKEND`（xpu 或 rocm），用来选 venv、`torch.xpu`/`torch.cuda`、`ZE_AFFINITY_MASK`/`HIP_VISIBLE_DEVICES`。新增 `JZL_EXTRA_ARGS`，内容追加到 `main.py` 参数后面。未提交。 |
| `~/comfyui/comfy/quant_ops.py` | 新增 `COMFY_CK_PRIORITY`，用逗号分隔的列表调用 `ck.set_backend_priority`。不设时行为不变。共 6 行，未提交。 |
| `~/h3/h3-b70-bench/common/gpu_sample_amd.py` | 新建。AMD 采样器，输出列与 `gpu_sample.py` 相同。利用率读 `gpu_busy_percent`；显存读 fdinfo `drm-resident-vram` 和 `mem_info_vram_used`；功耗读 hwmon `power1_average`；频率读 `freq1_input`。 |
| `~/h3/h3-b70-bench/common/run_one.py` | 按 `/sys/bus/pci/devices/0000:03:00.0/driver` 选采样器：xe 用原采样器，amdgpu 用 AMD 采样器。 |
| `common/smoke_int8_rocm.py`、`ck_backend_bench.py`、`ck_which_backend.py`、`stage_sum.py` | 新建。分别测 int8 三后端的速度和精度、测 DiT 真实形状、查实际选中的后端、按阶段拆分时间。 |
| `~/comfyui/.venv-rocm` | 新建 venv。torch 2.14.1+rocm7.14，torchvision 0.29.1+rocm7.14，torchaudio 2.11.0+rocm7.2。其余包与 `.venv` 相同，但不含 llama-cpp-python 和 Intel 运行时。 |
| 系统 | `apt upgrade` 升了 105 个包，其中 oneAPI 升到 2026.2。新装 `amdgpu-dkms` 31.60 和 `amdrocm10.1-gfx1201`。`dracut` 被换成 `initramfs-tools`。旧内核 7.0.0-34 保留。 |

复现方法：

```
cd ~/h3/h3-b70-bench/comfyui
export JZL_GPU_BACKEND=rocm          # 可选：JZL_EXTRA_ARGS=...  COMFY_CK_PRIORITY=triton,hip,eager
PY=~/comfyui/.venv-rocm/bin/python
$PY ../common/run_one.py . <跑次名> ~/comfyui -- $PY ~/h3/h3-b70-bench/comfyui/run.py 0 <跑次名>
```

## 数据位置

所有跑次的记录都在 `~/h3/h3-b70-bench/comfyui/runs/<跑次名>/`。原始输出和成片的位置：

| 代号 / 记录目录名 | 原始输出和成片 |
|---|---|
| B70：`comfyui-int8-lora-ra16m` | 记录目录下的 `output/` |
| R9700 默认：`comfyui-int8-lora-r9700` | `~/comfyui/output/shenbi-maliang/runs/comfyui-int8-lora-r9700_20261023-132353/` |
| R9700 triton：`comfyui-int8-lora-r9700-triton` | `…/comfyui-int8-lora-r9700-triton_20261023-133124/` |
| R9700 nodyn：`comfyui-int8-lora-r9700-nodyn` | `…/comfyui-int8-lora-r9700-nodyn_20261023-134047/` |
| R9700 ckt：`comfyui-int8-lora-r9700-ckt` | `…/comfyui-int8-lora-r9700-ckt_20261023-135357/` |

## 进行中和待办

1. **进行中**：herdr agent `aimdoxpu` 正在本机 B580 上实现 xpu 版 DynamicVRAM（comfy-aimdo）。配置是 pi，magpie DeepSeek-V4.1-Flash，thinking max。工作目录 `~/research/aimdo-xpu/`，任务书在该目录的 `TASK.md`。做完要回到 B70 上复测。
2. 查 R9700 nodyn 成片亮度为什么不同。
3. 提交 `run_shots.py` 和 `quant_ops.py` 的改动。
