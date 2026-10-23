# GR00T N1.7 三视角：优化路径复测

日期：2026-10-23。已实际完成 EmbodiInfer 完整优化路径、独立构建 FlashRT FP8 和 NVFP4 的串行测量。未接模拟器，未设置数值验收阈值。

## 结果

单位：ms。主指标为 GPU 输入就绪至 GPU 完整 action 完成的同步 CPU wall latency，包含 learned embeddings、backbone 和全部去噪。

| 路径 | mean | p50 | p95 | 平均 MAE | 最大绝对误差 | 最小 cosine |
|---|---:|---:|---:|---:|---:|---:|
| EmbodiInfer 前缀图 + Inductor + 完整去噪图 | 28.482 | 28.485 | 28.520 | 0.002731 | 0.160156 | 0.999957 |
| FlashRT 融合 FP8 / 混合 FP8-BF16 action head | 14.418 | 14.421 | 14.442 | 0.027798 | 1.300781 | 0.997901 |
| FlashRT FP8 backbone / NVFP4 DiT 与 cross-K/V | 8.974 | 8.976 | 9.014 | 0.028182 | 1.343750 | 0.998962 |

MAE 是 4 个不同 evaluation observation 的 MAE 均值，不重复计入 A/B/A 最后一条；误差相对冻结的官方 eager BF16 输出，比较完整 `[1,40,132]` 归一化动作。cosine 较高不能抵消较大的单元素误差。所有 `accuracy_gate=not_configured`，不宣称精度通过，也不推断任务成功率。

与 [首轮实测](../2026-10-23-5090-smoke/README.md) 比较：

| 路径 | 原 mean | 新 mean | 延迟变化 |
|---|---:|---:|---|
| EmbodiInfer | 70.439 | 28.482 | 下降 59.6%，约 2.47× 加速 |
| FlashRT FP8 | 16.541 | 14.418 | 下降 12.8% |
| FlashRT NVFP4 对本轮 FP8 | 14.418 | 8.974 | 下降 37.8%，精度路径不同 |

EmbodiInfer 旧路径 MAE/max-abs 为 0.001795/0.126953，新路径为 0.002731/0.160156；优化没有保持逐位一致。FlashRT 旧路径 MAE/max-abs 为 0.027105/1.296875。新旧 FlashRT 使用不同提交、执行路径和构建，不能把全部提升归因于某一个 kernel。以上只比较本协议下已测配置，不是引擎通用性能上限。

首轮官方 eager BF16 mean=83.068 ms、PhyAI mean=26.773 ms，本轮未重跑这两条路径。

## 固定条件与边界

- 单卡 RTX 5090，B=1，三视角 256×256，4 步去噪，动作块 40，输出 132 维。
- 同一个 base checkpoint `nvidia/GR00T-N1.7-3B`，revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`；不是 LIBERO 微调 checkpoint。
- 同一冻结数据集 `/data/inference-baseline/datasets/groot-n17-v3`，manifest SHA256：`562a86fc238ca10adcc2b623be09bf796b74f1403fbdd5ab6f6d1785335c2297`。
- `pixel_values=[768,1536] BF16`、token IDs/mask `[1,217]`、grid `[3,3]`、state `[1,1,132] BF16`、embodiment 24；同一显式初始噪声 `[1,40,132] BF16`。
- 1 个校准样本、4 个合成评估样本；每组 10 次预热、100 次串行计时。CPU intra-op=8 / inter-op=1，TF32 关闭，未锁 GPU 时钟。
- 外部预处理、输入 H2D、输出 D2H、动作解码、加载、校准、编译和图捕获不计时。引擎内部 metadata 传输、设备内 buffer copy、CPU 调度与同步仍计时。
- FlashRT 每次在适配器内计算 patch/token embedding，构造 fresh aux 后调用原生 infer；只缓存固定布局的 position/RoPE 等常量。
- 所有正式测量在 FlashRT 构建结束后执行，没有与源码编译或其他 GPU 推理并行。预先验证用 `embodiinfer-optimized-001` 不列入正式表。

## 前缀图与完整去噪图

前缀图把一次观测的条件编码捕获成 CUDA Graph：patch embedding → ViT/DeepStack → 语言编码 → VL/state 编码 → cross-attention K/V。每次请求都复制当前图像、文本和状态到固定输入缓冲区，再重放图；新观测的特征仍会重新计算。

完整去噪图把全部 4 步 `denoise_step`、Euler 更新和时间步推进一起捕获。每次请求更新前缀特征/KV 与初始噪声，然后一次重放整个循环。cross-K/V 在同一次请求的 4 步之间复用，不跨观测复用。

CUDA Graph 复用的是执行计划和缓冲区，减少逐 kernel 的 CPU 提交开销；Inductor 则编译并融合算子。两者是不同层次的优化，不意味着把模型结果缓存起来。

EmbodiInfer 使用上游 `Gr00tPolicy.encode_prefix`、`FlowDecoder.integrate` 和 `GraphManager(full_loop=True)`，`denoise_step` 按上游 benchmark 的方式使用 Inductor 编译，并关闭 Inductor 自己的 cudagraphs，由引擎负责捕获。上游源码未修改。运行记录确认：前缀图 1 个、完整去噪图 1 个、Inductor 编译图 1 个，预热后与正式测量后数量一致。

## FlashRT 独立构建与实际精度

- 源码：`/data/inference-baseline/third_party/FlashRT`，重新从 GitHub 克隆，固定提交 `1fbe995089af4148bacd10ead61764d0aeb80f7a`。
- 环境：该目录的 `.venv`，Python 3.12.14、Torch 2.11.0+cu128、Transformers 4.55.4。51 个安装包的依赖一致性检查通过；[完整依赖](flashrt-environment.txt)。
- CUDA toolkit：`/usr/local/cuda` 的 CUDA 13.0；CUTLASS v4.4.2，commit `da5e086dab31d63815acafdac9a9c5893b1c69e2`；CMake `GPU_ARCH=120`，关闭与此模型无关的 Motus kernels，其余采用上游默认构建。见 [构建配置](flashrt-build-config.txt)。
- `flash_rt_kernels` 和 `flash_rt_fa2` 均来自新目录；实际加载路径和 SHA256 记录在两个 FlashRT summary 的 `engine.binaries`。未使用 `/data/5090-machine/FlashRT` 的源码、环境或二进制。
- 独立校准缓存位于新源码目录 `.cache/baseline-v3-calibration`，不用旧仓库/用户默认校准缓存。初始化种子为 0；上游 warmup 使用 seed=0 的噪声和零状态。正式推理仍显式使用共享 fixture 噪声和当前 state。
- FP8 路径实际为融合 FP8 backbone；action head 的部分 FFN/QKV 也为 FP8，其余操作混合 BF16。不能描述为“整个 action head 纯 BF16”。
- NVFP4 路径保留融合 FP8 backbone，DiT 和 cross-K/V 采用 NVFP4，其他操作混合 BF16。记录确认 `dit_quant=nvfp4_skinny`、`nvfp4_cross_kv=true`。继承的 `dit_use_fp8=true` 标志仍存在，但由 `dit_quant` 分支优先选择 NVFP4，不代表 DiT 回退到 FP8。
- 两条路径均确认 backbone/action CUDA Graph 存在，捕获去噪步数为 4。新版融合 backbone 按上游实现执行到最后一个 DeepStack tap（18 个 ViT block），没有在本仓库额外修改模型计算。

安装过程中曾因服务器代理不可达而重试依赖下载；编译后的检查也曾错误要求导入 SM100/110 专用 `flash_rt_fp4`。现已修正安装脚本：SM120 GR00T NVFP4 所需符号位于主 kernel 库，并通过真实 NVFP4 运行验证。失败日志保留，未混入有效测量。

## 校验与记录

三条路径均通过以下结构/执行检查：输出 shape/device 正确且有限；A/B/A 重复输入稳定；分别只更换图像、state 或 noise 时输出发生变化；恢复输入后输出恢复；原始输入/noise 未被修改。它们不是任务精度验收，也没有覆盖任意 prompt/layout。

已有 7 项核心与 fixture 测试通过。逐调用均值、p50/p95/p99 已独立复算，确认 100 条记录、与官方 reference 相同的数据集 hash、预热后运行图状态不变；见 [核验记录](validation.json)。

- [汇总](comparison.md)
- [EmbodiInfer](embodiinfer_optimized.json)，服务器 `artifacts/embodiinfer-optimized-002`
- [FlashRT FP8](flashrt_fp8.json)，服务器 `artifacts/flashrt-fp8-fresh-001`
- [FlashRT NVFP4](flashrt_nvfp4.json)，服务器 `artifacts/flashrt-nvfp4-fresh-001`
- 同名 `*-latency.jsonl` 保留全部逐调用延迟。上游跟踪源码未修改；新 FlashRT 目录额外有生成的 `environment-freeze.txt`，非源码变更。

复跑使用新输出目录：

```bash
cd /data/inference-baseline
bash scripts/run_5090.sh embodiinfer artifacts/embodiinfer-optimized-next
bash scripts/run_5090.sh flashrt artifacts/flashrt-fp8-next
bash scripts/run_5090.sh flashrt_nvfp4 artifacts/flashrt-nvfp4-next
```

## 与上游公开数字的关系

FlashRT 官方报告双视角 GPU median：FP8 14.01 ms、NVFP4 8.64 ms。我们三视角同步 wall p50 分别为 14.421/8.976 ms，数值接近，但不能称为同条件复现或据差额量化多一个视角的成本。[上游说明](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md)

EmbodiInfer 官方优化版在 RTX 4090、双视角 LIBERO checkpoint 上报告 Forward mean 24.17 ms；我们为 RTX 5090、三视角 base checkpoint，且 Torch 版本不同，mean 28.482 ms。此次已补齐前缀图、Inductor 与完整去噪图，但仍不是同一实验条件。[上游 benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md)

本轮仍是小规模合成输入验证；100 次调用重复 4 个样本，不能给出有代表性的尾延迟、真实机器人动作质量或量化精度结论。
