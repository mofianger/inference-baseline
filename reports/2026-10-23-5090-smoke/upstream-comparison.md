# GR00T N1.7：上游公开结果与本地实测对照

核对日期：2026-10-23。本次只核对公开资料和已有运行记录，没有重新测量。上游数据为项目维护者发布的结果，未在我们的机器上复现。资料取自当日默认分支，后续可能变化。

## 结论与数值

FlashRT、PhyAI 发布了 RTX 5090 数据；EmbodiInfer 的 GR00T benchmark 提供 RTX 4090 和 Thor 数据，未在本次核对资料中找到 5090 结果。没有一组已确认与我们三视角协议完全相同，因此不计算跨协议加速比，也不据此判断性能回退。

| 引擎/配置 | 上游硬件 | 上游公开延迟 | 我们的 5090 mean / p50 | 可比性 |
|---|---|---|---|---|
| FlashRT FP8 | RTX 5090 | 14.01 ms，GPU time median；历史未融合版本 16.51 ms | 16.541 / 16.540 ms | 双视角 vs 三视角；GPU 时间 vs 同步 wall time；版本不同 |
| FlashRT NVFP4 action head + FP8 backbone | RTX 5090 | 8.64 ms，GPU time median | 未测此精度路径 | 不能用我们的 FP8/BF16 路径代表 NVFP4 |
| PhyAI | RTX 5090 | 20.6 ms，README 图表 | 26.773 / 26.753 ms | 图表未充分交代 checkpoint、视角数、统计量和计时边界 |
| EmbodiInfer，原生完整优化路径 | RTX 4090 | 24.17 ms，完整 Forward mean；31.30 ms，E2E mean | 70.439 / 66.626 ms | 硬件、checkpoint、视角、软件环境、优化开关均不同 |
| EmbodiInfer，旧基线 | Thor | 80.50 ms，纯推理 mean；92.47 ms，E2E mean | 未测 Thor | 不跨硬件计算倍数 |

来源：[FlashRT 测量与精度说明](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md)、[PhyAI README 性能图](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg)、[EmbodiInfer GR00T benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md)。

## 我们的共同协议

- 单卡 RTX 5090；GR00T-N1.7-3B base，revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`，不是 LIBERO 微调权重。
- B=1，三个 256×256 视角，4 步去噪，动作块 40，输出完整归一化 `[1,40,132]`。
- 预处理后的 GPU tensor 就绪到 GPU action 完成，使用两端 CUDA 同步的 CPU wall time。包括 learned patch/token embedding、backbone、全部去噪及内部调度；排除外部预处理、输入/输出 staging、动作反归一化。
- FlashRT 的动态 patch/token embedding 在计时区间内；固定布局的 position/RoPE 常量允许在初始化阶段缓存。不能直接把原生 `infer(aux=...)` 的计时等同于我们的适配器计时。
- 5 个合成 fixture：1 个校准，4 个评估；相同输入与初始噪声。10 次预热，100 次计时。CPU intra-op=8、inter-op=1，未锁 GPU 时钟。
- 本轮是接入验证，样本不足以代表真实任务分布或可靠的尾延迟。精度阈值未设置，只报告误差。

原始结果与限制见 [README](README.md)、[汇总](comparison.md) 及各引擎 JSON。

## FlashRT：接近历史数字，不等于复现

上游以 12 条真实 DROID 双视角 observation，4 步去噪，200 次 GPU 计时的中位数报告性能。每次传 fresh `aux`，包含 backbone graph 和 action graph。当前 FP8 的 14.01 ms 使用融合 backbone，并只执行到最后一个 DeepStack tap 的 18 个 ViT block；文档将 16.51 ms 标为执行全部 24 层、未融合的历史路径。NVFP4 的 8.64 ms 将 action head 与 cross K/V 改为 NVFP4，backbone 保留 FP8。

我们运行固定提交 `a7d57c1` 的 FP8 backbone / mixed BF16 action head，结果 p50=16.540 ms。它与历史 16.51 ms 数值接近，但视角数、输入分布、计时边界、版本不一致，不能称为复现成功。也不能把与当前 14.01 ms 的差额全部归因为多一个视角或 CPU 开销，需要分段实测。

上游 FP8 对 FP16 frontend 的归一化动作 mean/min cosine 为 0.99979/0.99811；对官方 policy 的 relative action 为 0.995/0.979。我们的比较对象是官方 eager BF16，比较完整归一化 132 维动作，min cosine=0.998504，最大绝对误差=1.296875。这些质量指标的参考实现和动作空间不同，不能直接排名；高 cosine 也不能消除大单元素误差。

## EmbodiInfer：当前接入没有覆盖其最佳优化路径

上游 2026-09-09 的 RTX 4090 最佳结果为完整 Forward mean=24.17 ms、E2E mean=31.30 ms。条件为 LIBERO-10 的 1,600 条真实观测、双视角 256px、B=1、BF16、4 步去噪；内部动作 `40×132`，CPU 物理输出 `16×7`。checkpoint 为 `nvidia/GR00T-N1.7-LIBERO/libero_10`，revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`。

完整 Forward 覆盖视觉/文本条件编码和全部去噪；E2E 还包含 CPU 预处理、噪声生成、传输、动作后处理。比较 model-only 应优先看 24.17 ms，不能直接拿 31.30 ms E2E 当成同一计时边界。

其最佳路径开启前缀 CUDA Graph、完整去噪 Graph、Inductor、跨去噪步前缀 K/V 复用，跳过无用 logits。运行环境为 Torch 2.13.0+cu129 / Transformers 5.3.0 / Diffusers 0.35.2，CPU 数值库线程为 1。

我们的 `Gr00tPolicy.sample_actions` 设置为 `native_inference=True`、`prefix_cuda_graph=False`，没有完整 denoise graph，Torch 2.10.0 / Transformers 5.3.0 / Diffusers 0.35.1。因此目前 mean=70.439 ms 只代表已接入路径，不能代表 EmbodiInfer 最优性能。应优先补齐这条优化路径，在现有三视角协议下重新验证误差并测量。

上游声称 10 条完整动作回归相对其原生实现逐位一致；这不是相对我们官方 eager baseline 的一致性证明。我们当前最大绝对误差为 0.126953。

上游还保留 Thor 旧路径 80.50 ms 纯推理 / 92.47 ms E2E，新优化尚未在 Thor 验证。README 中由 EmbodiInfer 作者复测的 PhyAI 数字属于第三方测量，不应称为 PhyAI 官方发布结果。

## PhyAI：有同 GPU 数字，历史测量条件不足

README 的性能图给出 GR00T N1.7：RTX 5090 上 PhyAI 20.6 ms、图中的 official baseline 47.0 ms；Thor 上 PhyAI 95.4 ms。图表没有充分交代这些数字对应的 checkpoint revision、视角数、输入长度、噪声、统计量及计时边界，无法确认与我们的三视角 base checkpoint 测试等价。

我们的固定提交 `f983a8f`、BF16/native FP32 norms、backbone/action CUDA Graph 路径 mean=26.773 ms、p50=26.753 ms，数值高于图中 20.6 ms，但不足以判断引擎回退或适配器引入了多少开销。同理，图中 official 47.0 ms 不能直接与我们官方 eager mean=83.068 ms 比较。

当前 [单 GPU 文档](https://github.com/mingti-org/phyai/blob/main/docs/models/gr00t/ws1.mdx) 的示例使用 LIBERO `libero_10` 权重；在初始化时捕获图，计时 `engine.step()`，排除预处理、GPU 输入传输、动作解码，3 次不计时加 30 次计时，打印多个统计量。它说明了当前示例协议，但未证明 README 历史图使用同一协议，不能反向补全图中未知条件。

我们当前最大绝对误差=0.566406、min cosine=0.999787；没有配置通过阈值，不能宣称精度通过。

## 后续验证顺序

1. 保持现有三视角协议，补齐 EmbodiInfer 的完整优化入口，先核验输入动态性及数值误差，再测延迟。
2. 为 FlashRT 增加当前融合 FP8 和 NVFP4 独立配置，保留旧提交结果；不能在同名结果下悄悄替换精度路径。
3. 如需复现上游数字，另建双视角真实 DROID/LIBERO 协议，并记录官方同样的 checkpoint、输入形状和精度，不修改现有三视角协议。
4. 同时记录同步 wall time 与完整模型区间 CUDA event time，明确各自覆盖的工作；对 PhyAI 图表仍未知的条件保持“未知”，不据当前示例猜测。

本次不修改引擎配置、不重跑测试；以上是根据对照发现的后续工作。
