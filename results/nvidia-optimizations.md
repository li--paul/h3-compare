# NVIDIA H3 / ComfyUI 优化对比（RTX PRO 4500 Blackwell 32 GB）

统一条件：REF2VA / 1376×768 / 124 帧 / 8 步 / CFG 1.0 / res_multistep+beta / seed 1001–1003。
基线为 INT8 ConvRot + PyTorch attention；其余只改运行时/内核，或明确标注换量化。

## 总表

| 配置 | 类型 | 单步 | 8 步采样 | 端到端 | 峰值显存 | 功耗（峰/均） | 相对基线 | PSNR / SSIM | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 baseline | same-quant | 23.43 s | 185.3 s | **216.6 s** | 31.1 GB | 203.4/198.0 W | 1.00× | — | PyTorch attention, default |
| INT8 + SOL sparse tau=1.3 | same-quant | 9.80 s | 84.1 s | **115.3 s** | 31.3 GB | 203.3/195.9 W | 2.39× | 18.013400 dB / 0.713275 | BlockSparseAttention, tau=1.3, start=0.0 |
| INT8 + SOL tau=1.0 | same-quant | 10.63 s | 89.4 s | **117.4 s** | 31.1 GB | 203.1/196.3 W | 2.20× | 18.226643 dB / 0.718433 | less sparse, higher quality |
| INT8 + SOL tau=2.0 | same-quant | 9.09 s | 76.3 s | **104.5 s** | 31.3 GB | 203.5/198.0 W | 2.58× | 14.317285 dB / 0.600363 | more sparse, faster/lower quality |
| INT8 + SOL + fast | same-quant | 10.07 s | 84.2 s | **112.7 s** | 31.1 GB | 203.2/197.3 W | 2.33× | 18.249442 dB / 0.723769 | SOL + --fast cublas_ops autotune |
| INT8 + SOL + Sage2.2 | same-quant | 10.06 s | 84.0 s | **112.3 s** | 31.3 GB | 202.4/197.3 W | 2.33× | 18.524380 dB / 0.723180 | SOL + --use-sage-attention (compat check) |
| INT8 + CK attention | same-quant | 16.57 s | 131.4 s | **162.3 s** | 31.2 GB | 203.2/198.7 W | 1.41× | 23.052139 dB / 0.813830 | --use-ck-attention |
| INT8 + SageAttention 2.2 | same-quant | 16.42 s | 130.5 s | **161.1 s** | 31.1 GB | 203.6/198.8 W | 1.43× | 21.457236 dB / 0.803908 | --use-sage-attention |
| INT8 + fast(cublas/autotune) | same-quant | 23.17 s | 185.1 s | **217.5 s** | 31.2 GB | 203.5/199.2 W | 1.01× | 21.802913 dB / 0.802606 | --fast cublas_ops autotune |
| INT8 + async-offload 4 | same-quant | 24.13 s | 190.0 s | **222.3 s** | 31.2 GB | 203.7/199.0 W | 0.97× | — | --async-offload 4 |
| W6A8 (official) | quant-variant | 24.51 s | 195.7 s | **226.3 s** | 31.1 GB | 203.6/198.9 W | 0.96× | 20.421823 dB / 0.788893 | 6-bit weights, int8 compute |
| FP8 scaled (official) | quant-variant | 26.19 s | 206.8 s | **239.5 s** | 31.1 GB | 202.6/198.8 W | 0.89× | 20.767536 dB / 0.757534 | --fast fp8_matrix_mult |

> 相对基线 = 基线单步 / 本配置单步；>1 表示更快。PSNR/SSIM 是对 seed 1001 输出与基线的逐帧比较。
> `same-quant` 行保持 INT8 ConvRot；`quant-variant` 行改变了量化格式，不能与 INT8 直接等同。

## 分项

- **INT8 baseline**：23.43 s/步，185.3 s 采样，216.6 s 端到端，4.9 s 条件编码，20.1 s VAE。
- **INT8 + SOL sparse tau=1.3**：9.80 s/步，84.1 s 采样，115.3 s 端到端，4.9 s 条件编码，20.8 s VAE。
- **INT8 + SOL tau=1.0**：10.63 s/步，89.4 s 采样，117.4 s 端到端，5.0 s 条件编码，20.7 s VAE。
- **INT8 + SOL tau=2.0**：9.09 s/步，76.3 s 采样，104.5 s 端到端，4.9 s 条件编码，20.7 s VAE。
- **INT8 + SOL + fast**：10.07 s/步，84.2 s 采样，112.7 s 端到端，5.3 s 条件编码，20.6 s VAE。 启动参数：`--fast cublas_ops autotune`。
- **INT8 + SOL + Sage2.2**：10.06 s/步，84.0 s 采样，112.3 s 端到端，4.9 s 条件编码，20.6 s VAE。 启动参数：`--use-sage-attention`。
- **INT8 + CK attention**：16.57 s/步，131.4 s 采样，162.3 s 端到端，5.0 s 条件编码，20.5 s VAE。 启动参数：`--use-ck-attention`。
- **INT8 + SageAttention 2.2**：16.42 s/步，130.5 s 采样，161.1 s 端到端，5.0 s 条件编码，20.4 s VAE。 启动参数：`--use-sage-attention`。
- **INT8 + fast(cublas/autotune)**：23.17 s/步，185.1 s 采样，217.5 s 端到端，5.5 s 条件编码，20.5 s VAE。 启动参数：`--fast cublas_ops autotune`。
- **INT8 + async-offload 4**：24.13 s/步，190.0 s 采样，222.3 s 端到端，5.0 s 条件编码，20.7 s VAE。 启动参数：`--async-offload 4`。
- **W6A8 (official)**：24.51 s/步，195.7 s 采样，226.3 s 端到端，4.8 s 条件编码，20.6 s VAE。
- **FP8 scaled (official)**：26.19 s/步，206.8 s 采样，239.5 s 端到端，4.9 s 条件编码，20.6 s VAE。 启动参数：`--fast fp8_matrix_mult`。

## 推荐

- **同量化最快：SOL tau=1.3**，9.80 s/步 / 115.3 s，2.39×；PSNR 18.0 dB、SSIM 0.713，画面结构一致但采样轨迹不同。
- **画质优先：CK attention 或 SageAttention 2.2**，约 1.4×；CK 的 PSNR 23.1 dB / SSIM 0.814 更好，Sage 略快（16.42 vs 16.57 s/步）。
- **SOL tau=2.0 不推荐**：9.09 s/步 / 104.5 s 虽然最快，但 PSNR 14.3 dB / SSIM 0.600，构图已明显偏离。
- **SOL + Sage / SOL + fast 不叠加收益**：SOL 已接管 attention，额外 attention 后端不再加速；`--fast` 反而略慢。
- **W6A8 / FP8 在 32 GB 上更慢**：24.5 / 26.2 s/步；27 GB 文本编码器仍占满显存，换量化没有带来装载优势。
- **`--fast cublas_ops autotune` 不建议**：速度 1.01×，但同 seed 输出已变化（PSNR 21.8 dB），属于数值路径改变。

## 口径与限制

- 所有 `same-quant` 行使用同一份 INT8 ConvRot 权重；只有 kernel/attention/offload 变化。
- `W6A8` 是 6-bit 权重 + int8 计算，`FP8` 是 FP8 scaled；两者都改变了量化格式，画质需单独看 PSNR/抽帧。
- SOL 会改变采样轨迹（不是逐像素等价），PSNR 下降是预期的；看画面结构而不是 PSNR 绝对值。
- SageAttention 需要 2.2.0；PyPI 只有 1.0.6，本轮从 `thu-ml/SageAttention@v2.2.0` 源码编译。
- 32 GB 显存下 FP8/W6A8 的换入行为由 comfy-aimdo 决定，峰值显存和端到端时间受装载影响。

