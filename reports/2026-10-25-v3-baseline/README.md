# V3 推理引擎 Baseline 总表

本页是历史性能资料归档。下一轮任务以 [5090 × GR00T N1.7 待测 Recipe 清单](../../recipes/groot-n17-5090-candidates.md) 为准，包含没有延迟结果的新候选。

整理日期：2026-10-25；原 HTML 自述核查日期：2026-10-23。本次是资料整理和现有结果归档，没有启动新的 GPU 测试。

**优先入口：[RTX 5090 × GR00T N1.7：FP8 / NVFP4 Baseline](5090-groot-n17.md)。**

来源：[协作者 HTML](../../AI高速推理框架性能总表_V3.html)；结构化数据：[baseline-data.json](baseline-data.json)。保留全部原始记录 ID、指标、条件和链接；源文件 SHA256：`c4ca3f054e0b19159e74ceeae5f1352ae8a29e2e0ae828880b306137f32ba117`。

## 按仓库约定解释

Setting 用平台 × 模型 × 精度族筛选；真正可比较的 Baseline 还必须固定 checkpoint、协议、测试集和工作量。Recipe 描述分模块精度、执行、校准及硬件约束，Run 才是一次测量。FP8 与 NVFP4 是两个精度族，不代表整个模型所有算子都使用该精度。

`reported` 是协作者汇编的外部数字；`measured` 是本仓库有原始 Run 的测量；缺失的组合保持待调研。HTML 的 profile 是筛选标签，不能当作可运行 Recipe。本次每个外部记录保留独立 recipe_candidate_id，避免把 eager/graph/fusion 或不同测量边界误合并；它们不是 41 个已验证配方。

原 HTML 的 official 标色不一定代表被测引擎自己发布：例如 PhyAI/vla.cpp/Embodied.cpp 的部分记录来自 EmbodiInfer benchmark。JSON 同时保留原标色、发布者及是否与引擎仓库一致；其余外部行未逐一重新核查上游，不能称为本次验证。

## 平台与模型

| 维度 | 整理结果 |
|---|---|
| 性能表平台 | RTX 4090、RTX 5090；本仓库已有 Run 为 RTX 5090 |
| 有性能记录的模型 | Qwen3-8B、π0.5、GR00T N1.7；具体 checkpoint 不能仅凭模型名称判定相同 |
| 仅规划提及 | Llama：无型号/性能行；Thor：历史研究关注平台，本 V3 无性能行 |
| 精度 | BF16/FP32 混合、FP8、INT8 W8A8、AWQ W4A16、GGUF Q8_0、NVFP4 混合；INT8/Q8_0 不归为 FP8，AWQ 不归为 NVFP4 |

## 引擎与 Recipe 范围

以下模型和平台列仅表示 V3 有相应记录，不能扩展为框架全部支持范围。APXInf/APXinf-robo 按原表作为一个家族，封装与底层仍需分别固定版本。

| 引擎 | 一句话描述 | 有记录模型 | 有记录平台 | 有记录精度 / Recipe 线索 |
|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGML/GGUF 驱动的本地 LLM 推理框架。 | Qwen3-8B | RTX 5090 | GGUF BF16；GGUF Q8_0 |
| [vLLM](https://github.com/vllm-project/vllm) | 以请求调度和 KV 管理为核心的 LLM 服务引擎。 | Qwen3-8B | RTX 4090、RTX 5090 | AWQ W4A16（FP16 激活）；BF16；FP8（block128） |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA GPU 上的 LLM 优化与部署引擎。 | Qwen3-8B | RTX 5090 | BF16；KV：BF16；FP8（per-tensor）；KV：FP8；NVFP4 W4A4 + BF16 保留模块；KV：FP8 |
| [FlashRT](https://github.com/flashrt-project/FlashRT) | 面向固定模型和硬件，以融合低精度 kernel 和 CUDA Graph 执行推理。 | GR00T N1.7、Qwen3-8B、pi0.5 | RTX 4090、RTX 5090 | Decode：NVFP4 W4A4；Prefill：W4A16；BF16 head；FP8 + BF16 attention；FP8 prefix + FP8 decoder；NVFP4 DiT／cross K/V + FP8 VLM；部分 attention 为 BF16；NVFP4 prefix + FP8 decoder；NVFP4 prefix／decoder + FP8 attention |
| [Realtime-VLA（含 V2）](https://github.com/dexmal/realtime-vla) | 以固定形状 Triton 算子和 CUDA Graph 加速 VLA。 | pi0.5 | RTX 4090、RTX 5090 | BF16 |
| [vla.cpp](https://github.com/VinRobotics/vla.cpp) | 将 VLA 封装为 GGUF 并通过 C++/GGML 执行。 | GR00T N1.7、pi0.5 | RTX 4090 | BF16 GGUF + FP32 激活；BF16／FP32 权重 + FP32 激活 |
| [Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp) | 以 C++/GGML 执行具身模型，并桥接 llama.cpp 骨干。 | GR00T N1.7、pi0.5 | RTX 4090 | BF16／FP32 原生混合；BF16／FP32 权重 + FP32 激活 |
| [PhyAI](https://github.com/mingti-org/phyai) | 通过模型 Runner、图捕获和 kernel policy 提供低延迟具身推理。 | GR00T N1.7、pi0.5 | RTX 4090、RTX 5090 | BF16 |
| [APXInf／APXinf-robo](https://github.com/infinigence/ApxInf) | 将准备阶段与执行阶段分离的推理栈；APXinf-robo 为相关封装。 | pi0.5 | RTX 4090 | BF16；INT8 W8A8 |
| [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) | 结合原生模型、编译、CUDA Graph 与去噪状态管理的具身引擎。 | GR00T N1.7、pi0.5 | RTX 4090 | BF16；BF16（原生优化）；BF16／FP32 原生混合；FP8（Triton） |

各框架的方法名称与解释、逐项来源完整保存在 JSON 的 engines.methods。方法清单只是能力线索，不表示每个测速配置都启用了这些技术。

## 外部 Baseline：完整 41 条记录

保留原指标：LLM 的吞吐/TTFT、具身模型的完整调用、GPU 图重放不能混排。并发 16 和 1 QPS 也不自动等于单请求 B1。空缺条件不补猜；不得据本表直接计算跨行加速比。

| ID | 平台 | 模型 | 引擎 | 精度 / Recipe 变体 | 原始指标 | 输入 / 边界 / 统计条件 |
|---|---|---|---|---|---|---|
| [M001](https://github.com/cybbbbbbb/qwen3-vllm-inference/blob/main/README.md) | RTX 4090 | Qwen3-8B | vLLM | BF16 | 输出总吞吐 494.18 token/s；TTFT 580.38 ms | 并发 16；I/O=1024/256；均值 |
| [M002](https://github.com/cybbbbbbb/qwen3-vllm-inference/blob/main/README.md) | RTX 4090 | Qwen3-8B | vLLM | FP8（block128） | 输出总吞吐 627.53 token/s；TTFT 451.48 ms | 并发 16；I/O=1024/256；均值 |
| [M003](https://github.com/cybbbbbbb/qwen3-vllm-inference/blob/main/README.md) | RTX 4090 | Qwen3-8B | vLLM | AWQ W4A16（FP16 激活） | 输出总吞吐 835.97 token/s；TTFT 656.70 ms | 并发 16；I/O=1024/256；均值 |
| [B037](https://github.com/Dexmal/realtime-vla) | RTX 4090 | pi0.5 | Realtime-VLA（含 V2） | BF16 | 模型／图执行中位数 38.90 ms | 三视角；224×224；B=1 |
| [r030](https://github.com/dexmal/realtime-vla) | RTX 4090 | pi0.5 | Realtime-VLA（含 V2） | BF16 | 模型推理 29.2 ms | 双视角；B=1 |
| [B015](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | RTX 4090 | pi0.5 | PhyAI | BF16 | 完整本地调用均值 41.31 ms | 双视角；10 步；B=1；数值核验 9/10 |
| [r022](https://github.com/infinigence/ApxInf) | RTX 4090 | pi0.5 | APXInf／APXinf-robo | BF16 | Graph replay P50 31.38 ms | 双视角；10 步；H=10；B=1 |
| [B012](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | RTX 4090 | pi0.5 | EmbodiInfer | BF16（原生优化） | 完整本地调用均值 38.89 ms | 双视角；10 步；B=1 |
| [B022](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) | RTX 4090 | pi0.5 | EmbodiInfer | BF16 | 完整本地调用均值 39.77 ms | 双视角；10 步；H=50；量化专项参照 |
| [B013](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | RTX 4090 | pi0.5 | vla.cpp | BF16／FP32 权重 + FP32 激活 | 完整本地调用均值 79.36 ms | 双视角；10 步；B=1 |
| [B014](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | RTX 4090 | pi0.5 | Embodied.cpp | BF16／FP32 权重 + FP32 激活 | 完整本地调用均值 89.99 ms | 双视角；10 步；B=1；已应用 correctness patch |
| [M012](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | RTX 4090 | pi0.5 | EmbodiInfer | BF16／FP32 原生混合 | 完整请求 P50 74.622／74.638 ms（两轮） | 三视角；10 步；B=1；RobotWin checkpoint |
| [B023](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) | RTX 4090 | pi0.5 | EmbodiInfer | FP8（Triton） | 完整本地调用均值 121.76 ms | 双视角；10 步；H=50；关闭 Inductor |
| [B067](https://github.com/RLinf/APXinf-robo) | RTX 4090 | pi0.5 | APXInf／APXinf-robo | INT8 W8A8 | 模型／图执行 P50 25.99 ms | 双视角；10 步；H=10；服务封装公布值 |
| [r023](https://github.com/infinigence/ApxInf) | RTX 4090 | pi0.5 | APXInf／APXinf-robo | INT8 W8A8 | Graph replay P50 21.92 ms | 双视角；10 步；H=10；B=1 |
| [N029](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | RTX 4090 | GR00T N1.7 | PhyAI | BF16 | 完整本地调用均值 33.97 ms | 双视角；4 步；B=1；数值核验 9/10 |
| [B016](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | RTX 4090 | GR00T N1.7 | EmbodiInfer | BF16（原生优化） | 完整本地调用均值 31.30 ms | 双视角；4 步；B=1 |
| [B017](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | RTX 4090 | GR00T N1.7 | vla.cpp | BF16 GGUF + FP32 激活 | 完整本地调用均值 38.70 ms | 双视角；4 步；B=1 |
| [B018](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | RTX 4090 | GR00T N1.7 | Embodied.cpp | BF16／FP32 原生混合 | 完整本地调用均值 73.17 ms | 双视角；4 步；B=1 |
| [B009](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) | RTX 4090 | GR00T N1.7 | FlashRT | FP8 + BF16 attention | 模型／图执行 P50 32.60 ms | 双视角；B=1；eager；不含完整前后处理 |
| [B010](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) | RTX 4090 | GR00T N1.7 | FlashRT | FP8 + BF16 attention | Graph replay 均值 9.98 ms | 双视角；B=1；热态 Graph；仅 2 次采样 |
| [B011](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) | RTX 4090 | GR00T N1.7 | FlashRT | FP8 + BF16 attention | Graph replay 均值 9.69 ms | 双视角；B=1；FFN 融合；仅 4 次采样 |
| [M008](https://arxiv.org/html/2605.00519v1#S4.T4) | RTX 5090 | Qwen3-8B | llama.cpp | GGUF BF16 | Decode 92.2 token/s；TTFT 44.0 ms | 单请求；I/O≈496/256；3 次均值 |
| [M011](https://github.com/pegainfer-project/pegainfer/blob/main/docs/models/qwen3/serving-performance.md) | RTX 5090 | Qwen3-8B | vLLM | BF16 | 输出总吞吐 125.0 token/s；TTFT P50 97.4 ms | 1 QPS；I/O=1024/128；开启前缀缓存 |
| [M004](https://computingforgeeks.com/nvfp4-quantization-tensorrt-llm/) | RTX 5090 | Qwen3-8B | TensorRT-LLM | BF16；KV：BF16 | Decode 99.8 token/s | 单请求；I/O=472/256；流式测试 |
| [M005](https://computingforgeeks.com/nvfp4-quantization-tensorrt-llm/) | RTX 5090 | Qwen3-8B | TensorRT-LLM | FP8（per-tensor）；KV：FP8 | Decode 170.0 token/s | 单请求；I/O=472/256；流式测试 |
| [M009](https://github.com/kekzl/imp/blob/main/docs/BENCHMARKS.md) | RTX 5090 | Qwen3-8B | llama.cpp | GGUF Q8_0 | Decode 152.74 token/s | 合成基准；10 次重复；I/O 未公开 |
| [M010](https://github.com/kekzl/imp/blob/main/docs/BENCHMARKS.md) | RTX 5090 | Qwen3-8B | llama.cpp | GGUF Q8_0 | Prefill 14,258 token/s［13,644–14,286］ | pp512 独立测试；3 次中位数［范围］ |
| [M007](https://github.com/flashrt-project/FlashRT/blob/main/docs/qwen3_8b_nvfp4.md) | RTX 5090 | Qwen3-8B | FlashRT | Decode：NVFP4 W4A4；Prefill：W4A16；BF16 head | Decode 150 token/s；Graph TTFT 24.8 ms（P1024） | B=1；Graph 热态；独立 SFT checkpoint |
| [M006](https://computingforgeeks.com/nvfp4-quantization-tensorrt-llm/) | RTX 5090 | Qwen3-8B | TensorRT-LLM | NVFP4 W4A4 + BF16 保留模块；KV：FP8 | Decode 222.7 token/s | 单请求；I/O=472/256；原生 FP4 路径 |
| [B042](https://github.com/Dexmal/realtime-vla) | RTX 5090 | pi0.5 | Realtime-VLA（含 V2） | BF16 | 模型／图执行中位数 27.40 ms | 三视角；224×224；B=1 |
| [B041](https://github.com/Dexmal/realtime-vla) | RTX 5090 | pi0.5 | Realtime-VLA（含 V2） | BF16 | 模型／图执行中位数 21.30 ms | 双视角；224×224；B=1 |
| [B032](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg) | RTX 5090 | pi0.5 | PhyAI | BF16 | 单请求 28.60 ms | 视角、步数、计时范围与统计量未公开 |
| [M013](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) | RTX 5090 | pi0.5 | FlashRT | FP8 prefix + FP8 decoder | 同步 E2E P50 15.08 ms；P95 15.28 ms | 双视角；B=1；state 写入 prompt |
| [r036](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) | RTX 5090 | pi0.5 | FlashRT | FP8 prefix + FP8 decoder | 动作推理 P50 13.10 ms；P95 13.18 ms | 双视角；10 步；H=10；无 state token |
| [M015](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) | RTX 5090 | pi0.5 | FlashRT | NVFP4 prefix／decoder + FP8 attention | 同步 E2E P50 10.39 ms；P95 10.60 ms | 双视角；B=1；state 写入 prompt |
| [r037](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) | RTX 5090 | pi0.5 | FlashRT | NVFP4 prefix／decoder + FP8 attention | 动作推理 P50 8.75 ms；P95 8.78 ms | 双视角；10 步；H=10；无 state token |
| [M014](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) | RTX 5090 | pi0.5 | FlashRT | NVFP4 prefix + FP8 decoder | 同步 E2E P50 11.18 ms；P95 11.41 ms | 双视角；B=1；state 写入 prompt |
| [B034](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg) | RTX 5090 | GR00T N1.7 | PhyAI | BF16 | 单请求 20.60 ms | 视角、步数、计时范围与统计量未公开 |
| [B007](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md) | RTX 5090 | GR00T N1.7 | FlashRT | FP8 + BF16 attention | 模型／图执行中位数 14.01 ms | 双视角；4 步；B=1；GPU event 计时 |
| [B008](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md) | RTX 5090 | GR00T N1.7 | FlashRT | NVFP4 DiT／cross K/V + FP8 VLM；部分 attention 为 BF16 | 模型／图执行中位数 8.64 ms | 双视角；4 步；B=1；GPU event 计时 |

## 本仓库实测 Baseline

以下 11 条均来自 2026-10-23 已有 Run，不是本次重跑。只在相同模型协议与资产下比较；GR00T 的 reference/PhyAI 来自首轮，其他 GR00T 路径来自优化轮。旧 FlashRT 和旧 EmbodiInfer 不再作为当前 Baseline，历史记录仍保留。

| 模型 | 引擎 / 路径 | Mean ms | P50 ms | P95 ms | 平均 MAE | 原始 Run |
|---|---|---:|---:|---:|---:|---|
| GR00T N1.7 | flashrt_fp8 | 14.418 | 14.421 | 14.442 | 0.027798 | [summary](../../reports/2026-10-23-5090-optimized/flashrt_fp8.json) |
| GR00T N1.7 | flashrt_nvfp4 | 8.974 | 8.976 | 9.014 | 0.028182 | [summary](../../reports/2026-10-23-5090-optimized/flashrt_nvfp4.json) |
| GR00T N1.7 | embodiinfer_optimized | 28.482 | 28.485 | 28.520 | 0.002731 | [summary](../../reports/2026-10-23-5090-optimized/embodiinfer_optimized.json) |
| GR00T N1.7 | phyai | 26.773 | 26.753 | 26.934 | 0.003726 | [summary](../../reports/2026-10-23-5090-smoke/phyai.json) |
| GR00T N1.7 | reference | 83.068 | 82.964 | 85.449 | 0.000000 | [summary](../../reports/2026-10-23-5090-smoke/reference.json) |
| pi0.5 | EmbodiInfer pi05 full request graph | 91.255 | 91.245 | 91.328 | 0.003662 | [summary](../../reports/2026-10-23-5090-pi05/embodiinfer/summary.json) |
| pi0.5 | FlashRT pi05 BF16 full graph | 38.104 | 38.096 | 38.199 | 0.004639 | [summary](../../reports/2026-10-23-5090-pi05/flashrt-bf16/summary.json) |
| pi0.5 | FlashRT pi05 FP8 prefix and decoder | 22.517 | 22.515 | 22.548 | 0.025541 | [summary](../../reports/2026-10-23-5090-pi05/flashrt-fp8/summary.json) |
| pi0.5 | FlashRT pi05 NVFP4 prefix and decoder | 14.229 | 14.223 | 14.270 | 0.111926 | [summary](../../reports/2026-10-23-5090-pi05/flashrt-nvfp4/summary.json) |
| pi0.5 | PhyAI pi05 BF16 graphs | 41.773 | 41.851 | 41.906 | 0.004542 | [summary](../../reports/2026-10-23-5090-pi05/phyai/summary.json) |
| pi0.5 | OpenPI pi05 eager BF16/FP32 | 174.256 | 173.602 | 180.340 | 0.000000 | [summary](../../reports/2026-10-23-5090-pi05/reference/summary.json) |

全部 accuracy_gate=not_configured；误差不代表任务成功率。π0.5 的详细协议见 [运行指引](../../docs/guides/pi05-5090.md)，GR00T 重点分组见 [专项 Baseline](5090-groot-n17.md)。

## 维护与接入

候选配方先补齐 checkpoint/revision、硬件 kernel 支持、分模块 W/A/KV 精度、校准集、步数/H/A、计时端点与缓存规则，再新增 recipes 文档、configs 和 Adapter；实际运行完成后才标 measured。无法映射当前模型协议的候选应建立新协议，不能覆盖已有结果。

重新导入（只生成本总表和 JSON，不覆盖专项人工审查）：`python scripts/import_baseline_v3.py`。依赖 beautifulsoup4，仅用于资料导入，不加入 GPU 引擎环境。
