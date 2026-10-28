# AGENTS.md

## 仓库定位

- `nv_h3` **不是 git 仓库**，没有构建/测试/lint 体系。内容是：
  1. `MiniMax-H3_环境与模型版本清单.md` —— XPU 交付包的原始环境事实；
  2. 同一套 INT8 ConvRot / Ref2VA 配置在多张卡上的单卡基准与对比。
- 每张卡的结果按 `RUN_TAG` 归档在 `results/<tag>/`、`output/<tag>/`、`logs/<tag>/`；
  多卡总表是 `results/comparison.md`。不要手改生成物。
- `models/` 是 53 GB 权重，与 XPU 交付包同名同字节（`results/sha256.txt`）。不要删除或移动；重下约 50 GB。
- 运行说明见 `README.md`，本文件只记容易踩的坑。

## 运行

```bash
cd /home/acm/work/nv_h3
RUN_TAG=rtxpro4500 bash scripts/run_all.sh   # 下载 -> 准备 -> 启动 -> 冷+2热 -> 报告 -> 对比 -> 停止
RUN_TAG=rtxpro4500 bash scripts/03_serve.sh  # 单独起服务: 127.0.0.1:8190
RUN_TAG=rtxpro4500 bash scripts/04_bench.sh  # 需要服务已就绪
RUN_TAG=rtxpro4500 bash scripts/05_report.sh # 从 results/<tag>/timings.json 重生成报告
bash scripts/06_compare.sh                  # 生成 results/comparison.md / .csv
bash scripts/stop.sh
```

- 已有归档：`rtx6000ada`（RTX 6000 Ada）、`rtxpro4500`（RTX PRO 4500 Blackwell）。
  **换卡后必须用新 `RUN_TAG`**，否则会覆盖旧数据。
- ComfyUI 本体在仓库外：`/home/acm/work/test/comfy-h3/ComfyUI`（0.40.0，commit `e0a32bc`），
  venv 为 `/home/acm/work/test/comfy-h3/.venv`。**不要另建环境**：INT8 ConvRot 依赖
  comfy-kitchen 0.2.37 的 CUDA backend（已含 sm_120 cubin）。
- 数据路径全部通过 `config/extra_model_paths.yaml` 和 `--input/--output/--temp-directory`
  指回 `nv_h3/`；ComfyUI 自带配置不受影响。
- 每条命令追加到 `logs/commands.log`。排查顺序：
  `logs/commands.log` → `logs/<tag>/comfy_*.log` → `logs/<tag>/events/*.json`。
- 3 次运行约 11 分钟（两张 NVIDIA 卡都是这个量级）。

## 必须知道的坑

1. **autogrow 输入名带组前缀**：参考图要写成 `ref_images.ref_image_0`，不是 `ref_image_0`。
   写错会得到 `execute() got an unexpected keyword argument`。
2. **复跑必须同时换 seed 和提示词**：只换 seed 时 ComfyUI 仍会命中
   `MiniMaxH3ReferenceToVideo` 的条件缓存，测不到 TE/参考图编码。`bench.py` 已把
   `[Shot 1]` 改成 `[Shot N]` 来规避。
3. **几何/采样参数硬编码在工作流 JSON**（`workflows/ref2va_1376x768_8step.json` 的节点 6/7/8/10）。
   `WIDTH/HEIGHT` 环境变量只控制 `02_prepare.sh` 抽出的参考图尺寸，不会改工作流。
   改工作流后要同步 `scripts/05_report.py` / `06_compare.py` 里的参照常量。
4. **视频 VAE 用 fp16 文件 + `--bf16-vae`**：文档写的是 bf16，但 HF 只发布 fp16；让 ComfyUI
   以 bf16 运行即可，不存在名为 `..._video_vae_bf16.safetensors` 的下载。
5. **NVFP4 分卡**：Ada（sm_89）没有 NVFP4 张量核，`nvfp4`/`mxfp8` 会走 emulated；
   RTX PRO 4500（sm_120）上它们是 native。但为了量化口径一致，**本基准一律用 INT8**，
   不要用 `h3-comfy` 里的 nvfp4 文本编码器替代。
6. **显存**：Ada 48 GB 在 1376×768 / 124 帧下峰值 48.3 GB（98%）；
   PRO 4500 32 GB 峰值 31.1 GB，靠 comfy-aimdo 自动换入换出，未 OOM。
   不要随手加 batch、latent upscaler 或提高分辨率；32 GB 上尤其容易触发换页。
7. **算子路径不同**：CUDA 上 `int8_tensorwise` 是 native INT8 GEMM；XPU 上
   `supports_int8_compute()` 为 False，同一份权重反量化成 BF16。对外表述必须是
   “同权重、不同 kernel”，否则会把 kernel 优势误读成纯硬件差距。
   同 seed 下 Ada 与 PRO 4500 的输出画面结构一致但细节不同（PSNR≈19.7 dB / SSIM≈0.79），
   这是不同 kernel 的数值分歧，不是模型跑错；不要用逐像素一致做验收。
8. **冷启动磁盘是热的**：权重大概率仍在 page cache。要测真冷盘，先
   `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` 再跑。
9. **外部基准数字以 `results/sources/` 的实测报告为准**：
   - B60：`h3单卡测试总结_20261022.md`（653 s / 采样 563 s / 70 s 一步 / 20.2 GB / 181 W）。
   - B70、R9700：`report-r9700-vs-b70.md`（1344×768；B70 40.29 s/步，R9700 默认 44.92 s/步）。
   - B60 运行时分层：`h3-comfyui-xpu-perf.md`（stock 70.1 s/步；Intel 镜像 dense 46.4；SOL 36.97）。
     来源 <https://github.com/li--paul/h3-comfyui-xpu>。
   - **B60 功耗/显存有两套**：summary 20.2 GB / 181 W peak / 148 W avg；perf.md sysfs
     21.4 GB / 139 W mean / 164 W peak。采样方法不同，不可混算。
   不要引用环境清单 §10 的“单卡 10.8 min”，它与同节分项（2+10.5+1=13.5 min）矛盾。
10. **报告是生成物**：`results/<tag>/report.md` 由 `05_report.py` 生成，
    `results/comparison.md` 由 `06_compare.py` 生成。改脚本后重跑对应 `.sh`。
11. **32 GB 放不下时只改启动参数**：`EXTRA_ARGS="--cache-none"`，仍不行再 `--lowvram`；
    实际值写入 `logs/<tag>/run_flags.txt`，不要改分辨率/帧数/步数。
12. **报告里的卡名来自 `logs/<tag>/card.txt`**（运行时记录），不是当前 GPU。
    换卡后重新生成旧报告仍能保持正确；不要删这个文件。
13. **跨报告墙钟不可直接比较**：B70 478.3 s vs R9700 默认 448.3 s 的差距主要来自 DynamicVRAM
    （R9700 开、B70 没有）；关掉后 R9700 变成 690.1 s。比较算力请用“单步@1376”，
    外部 1344×768 数据乘 `1.0238`；比较能耗用“单步 × 忙碌功耗”。
14. **SOL 是 NVIDIA 侧最大收益**：`BlockSparseAttention` + `selection=sol-attn` + `tau=1.3` +
    `start_percent=0.0` + `sink_conditioning=exact_kv_and_rows`，9.8 s/步 / 2.39×。
    API 里动态组合用平铺键（`"selection": "sol-attn"`、`"selection.tau": 1.3`），不要嵌套。
    `verbose=true` 应打印 `sparse producer path: ... tokens`；tau=2.0 会掉到 PSNR 14.3，不推荐。
15. **SageAttention 2.2.0 不在 PyPI**：只有 1.0.6。需从 `thu-ml/SageAttention@v2.2.0` 源码编译：
    C++20、CUDA 13.1 编译，`cusparse.h`/`cublas_v2.h` 从 CUDA 12.8 头目录补进兼容 include；
    装完用 `--use-sage-attention`，实测 16.4 s/步 / 1.43×。
16. **`--fast cublas_ops autotune` 会改数值**：同 seed 输出与基线不同（PSNR 21.8 dB），
    速度只有 1.01×；不要把它当成“无损加速”。W6A8/FP8 在 32 GB 上也被 27 GB TE 拖慢，反而更慢。

## 已知结果（详见 results/comparison.md）

- B60 stock ComfyUI：70.1 s/步，652 s，12.7 kJ/步，26.8 Wh/段。
- B60 + Intel `llm-scaler-omni` 镜像 dense：46.4 s/步，470 s（native INT8 + CUTE，1.51×）。
- B60 + SOL（start=0.0）：36.97 s/步，370 s（1.90×，但改变采样轨迹，不是同配置）。
- RTX 6000 Ada：24.0 s/步，单段 223 s，7.1 kJ/步，18.4 Wh/段。
- RTX PRO 4500：23.4 s/步，单段 217 s，4.6 kJ/步，11.9 Wh/段。
- B70（外部，1344→1376）：41.2 s/步，322 s 去噪，13.4 kJ/步；墙钟 478 s（无 DynamicVRAM）。
- R9700 默认（外部）：46.0 s/步，359 s 去噪，13.5 kJ/步；墙钟 448 s（开 DynamicVRAM）。
- RTX PRO 4500 同量化优化：SOL tau=1.3 **9.8 s/步 / 115 s / 2.39×**（PSNR 18.0）；
  CK attention 16.6 s / 1.41×（PSNR 23.1）；Sage2.2 16.4 s / 1.43×（PSNR 21.5）。
  W6A8 24.5 s、FP8 26.2 s，在 32 GB 上反而更慢。
- 去噪排序（含运行时分层）：PRO 4500 ≈ Ada > B60+SOL > B70 > R9700 ≈ B60+dense > B60 stock。
  每步能耗：PRO 4500 < Ada < B60 < B70 ≈ R9700。

## 权重来源（重下时用）

- `Comfy-Org/MiniMax-H3`：`minimax_h3_ref2va_pruned_int8_convrot.safetensors`（20 970 379 616 B）、
  `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（27 141 342 152 B）、两个 VAE。
- `RunningHubAI/rh-minimax-h3-ref2v-turbo-8step-v1.0-768p-comfyui-bf16.safetensors-lora`：
  `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`（1 956 193 000 B）。

## 关键产物

- `results/<tag>/timings.json` —— 逐 run、逐 step 原始计时；`timings.csv` —— 汇总。
- `results/nvidia-optimizations.md` —— NVIDIA 运行时优化（SOL / CK / Sage / W6A8 / FP8）总表。
- `results/all-cards.md` —— Intel / AMD / NVIDIA 全卡一张表。
- `results/sources/` —— B60 实测总结、B70/R9700 对比报告（外部参考表的原始数据）。
- `logs/<tag>/nvidia-smi_*.csv` —— 1 Hz 显存/功耗/时钟；`comfy_*.log` —— 服务日志。
- `output/<tag>/bench/*.mp4` —— 1376×768 / 124 帧 / 24 fps / AAC 32 kHz 立体声。
