# GR00T N1.7 三视角：5090 接入验证

日期：2026-10-23（Asia/Shanghai）。本轮实际运行了官方 reference、FlashRT、EmbodiInfer、PhyAI，验证统一数据集、协议、runner 和本地/服务器 Git 协作。不接模拟器，不评估任务成功率。

## 固定条件

- 硬件：单张 NVIDIA GeForce RTX 5090，默认功耗限制 575 W，未锁频；具体驱动、频率与温度见各 summary。测试串行执行，无其他 GPU 模型任务。
- checkpoint：本机已有 `nvidia/GR00T-N1.7-3B`，revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`；模型文件 hash 随结果记录。不是 LIBERO finetuned checkpoint。
- B=1，三个不同视角，256×256；4 个去噪步；H=40，输出 `[1,40,132]` normalized action。
- 输入已经由官方 Cosmos processor 处理；`pixel_values=[768,1536] BF16`，token IDs/mask `[1,217] int64`，grid `[3,3] int64`，state `[1,1,132] BF16`，embodiment ID=24。
- 相同噪声 tensor，固定 prompt；5 个合成 fixture，首个只用于 FlashRT 校准，其余 4 个评估。数据集没有真实任务语义；state 也是合成归一化数值。
- 主计时：GPU tensor 就绪 → GPU action tensor 完成的同步 wall time。含全部 learned embeddings、backbone 和动作头，不含输入/输出 staging、原始图像处理与物理动作解码。
- 10 次预热，100 次串行计时；PyTorch intra-op threads=8、inter-op=1、TF32 disabled。引擎初始化可能改全局设置，Runner 随后恢复声明的控制参数。

## 结果

| 路径 | mean ms | p50 ms | p95 ms | 平均 MAE | 最大绝对误差 | 最小 cosine |
|---|---:|---:|---:|---:|---:|---:|
| 官方 eager SDPA BF16 | 83.068 | 82.964 | 85.449 | 0 | 0 | 1.000000 |
| FlashRT FP8 backbone / mixed BF16 action head | 16.541 | 16.540 | 16.571 | 0.027105 | 1.296875 | 0.998504 |
| EmbodiInfer native BF16 | 70.439 | 66.626 | 102.992 | 0.001795 | 0.126953 | 0.999976 |
| PhyAI BF16 / native FP32 norms | 26.773 | 26.753 | 26.934 | 0.003726 | 0.566406 | 0.999787 |

误差以官方参考 action 为基准，比较全部 132 维，不 mask 掉不使用的维度。平均 MAE 为 4 个不同 evaluation observation 的 MAE 平均值；最大误差/最小 cosine 取其中极值。A/B/A 最后一次重复调用不重复计入平均值。

**所有运行的 accuracy_gate 都是 `not_configured`。**用户当前要求只报告误差，未指定验收阈值。FlashRT 和 PhyAI 有较大的单元素误差；整体 cosine 高不代表这些误差可接受。现阶段不能宣称数值等价或任务精度通过。

## 路径与限制

- FlashRT 使用服务器已有提交 `a7d57c1`。适配器在每次计时内计算 BF16 patch/token embedding，并传入 fresh aux，确保不是复用旧视觉特征或仅测 action head。
- EmbodiInfer 使用提交 `2cd9a37` 的 `Gr00tPolicy.sample_actions`，native inference 开启，但 prefix graph 和完整 denoise graph 未启用。这是已验证的基础接入路径，不能代表该引擎的最佳延迟；本轮 p95 波动也较大，没有足够重复轮次归因。
- PhyAI 使用提交 `f983a8f`，backbone 和 action head 的 CUDA Graph 开启。HF snapshot 文件以同字节硬链接保留扩展名供其 loader 加载；上游源码未修改。
- 官方参考提交 `51d4c89` 的源代码未改；该已有工程的 `pyproject.toml`、`uv.lock` 在本轮开始前就有未提交改动。复用环境为 torch 2.9.0+cu128 / transformers 4.57.3；原始 dirty 状态保留在结果中。
- 各引擎使用独立依赖环境，Torch/Transformers 版本不同，结果不是“只改变一个内核实现”的受控实验。
- fixture 很小且是随机合成值，FlashRT 量化校准也仅使用一个合成样本。100 次循环调用仅用于接入验证，不足以给出可靠尾延迟或实际任务结论。
- 峰值显存为 PyTorch allocator 的 allocated/reserved，不包含所有第三方直接 CUDA 分配。
- 本次不同运行的 runner commit 不完全相同：期间增加了严格输入检查和引擎初始化后的全局参数恢复。最终各运行记录的实际 CPU threads 均为 8，TF32 均关闭，数据集 manifest hash 相同。

## 原始记录

- [汇总表](comparison.md)
- [固定数据集 manifest](dataset-manifest.json)：输入形状、dtype、样本 hash、checkpoint hash、参考环境。
- [reference](reference.json)、[FlashRT](flashrt.json)、[EmbodiInfer](embodiinfer.json)、[PhyAI](phyai.json)：完整配置、版本、逐样本/逐维误差、硬件状态和统计。
- 对应的 `*-latency.jsonl` 保留全部逐请求延迟和 sample ID。

大文件仍在服务器 `/data/inference-baseline/datasets/groot-n17-v3`；完整运行目录为 `artifacts/reference-001`、`artifacts/flashrt-001`、`artifacts/embodiinfer-001`、`artifacts/phyai-004`。中间失败运行保留为独立目录，不混入此报告。

下一步可先替换成真实观测预处理后的 tensor，再约定按动作维度的误差门槛；随后补齐 EmbodiInfer graph 路径、扩大数据与重复轮次。π0.5 尚未接入，本轮只实现了 GR00T N1.7。
