# π0.5 / GR00T N1.7 在 Thor、RTX 4090、RTX 5090 上的公开性能

调研日期：2026-10-22。共 74 条公开记录，其中 24 条有完整本地策略调用边界证据。全部为发布方/第三方报告，**本轮没有实机复测**。网页与默认分支可能继续更新；引用为访问时内容，除明确列出的提交外，不声称已固定完整源码版本。

## 阅读口径

- **E：完整本地策略调用**：已解码的 CPU observation → CPU 物理动作，包含相关预处理、模型推理及动作后处理。通常不包含摄像头采集、图像文件读取/解码、网络/RPC、执行器执行、模型加载、编译及 warmup；固定 prompt 预处理也可能在 setup，详见 profile。
- **M：模型/Graph**：输入已经处理或已在 GPU，仅测模型、predict、graph replay 等。不能称为完整端到端。
- **U：边界未充分确认**：原文只有 latency/total/E2E 标签，或分项统计求和，缺少可对齐的完整调用边界。数字保留但不纳入完整端到端排名。
- 每项延迟单位均为 **ms / 一次策略调用（一个 action chunk）**，不是 ms/action。动作块长度 H 不同、视角不同、去噪步数不同、mean/p50/min 不同，都不能直接排名。
- “精度”同时记录数值格式和可获得的质量证据。数值 cosine、数值 gate 及任务成功率是不同指标；没有闭环结果时不从数值一致性推断任务成功率。
- Thor 的功耗、锁频、JetPack 版本影响明显；未知项写明，不补猜。消费显卡未披露功耗/频率的记录也没有推定统一设置。

## 先看可用的端到端证据

| 模型 / 硬件 | 已找到的完整本地调用数据 | 对比限制 |
|---|---|---|
| π0.5 / Thor | FlashRT 20.33 NVFP4、40.71 FP8；EmbodiInfer BF16 98.23（专项）、164.05（历史） | FlashRT H10；EmbodiInfer 主 E2E H50，版本不同 |
| π0.5 / RTX 4090 | 同一 EmbodiInfer 对比：EmbodiInfer 38.89、PhyAI 41.31、vla.cpp 79.36、Embodied.cpp 89.99 | 同输入 H50；由 EmbodiInfer 发布，PhyAI 数值 gate 9/10 |
| π0.5 / RTX 5090 | FlashRT FP8 13.10、NVFP4 mixed 8.75 | H10；其他大量数字为模型-only，不能混排 |
| GR00T N1.7 / Thor | FlashRT mixed 27.89；官方 TensorRT mixed 35.9、BF16 77.6；EmbodiInfer BF16 92.47 | FlashRT 单相机；官方/EmbodiInfer 双相机；checkpoint/协议仍需对齐 |
| GR00T N1.7 / RTX 4090 | 同一 EmbodiInfer 对比：EmbodiInfer 31.30、PhyAI 34.48、vla.cpp 38.70、Embodied.cpp 73.17 | PhyAI 数值 gate 9/10；发布方为参赛引擎作者 |
| GR00T N1.7 / RTX 5090 | 本轮未找到边界充分明确的完整 E2E | FlashRT 8.64 是 GPU event median；vla.cpp 19.5 是合成输入 predict 最小值 |

以上数字的来源、精度和 profile 均在逐条记录表中；本表不构成跨配置榜单。

## 完整本地策略调用

| ID | 引擎 | 模型 | GPU | 数值格式 / 路径 | 延迟 ms | 统计 | 输入 profile | 备注 | 来源 |
|---|---|---|---|---|---:|---|---|---|---|
| B001 | FlashRT | π0.5 | Thor | FP8 | 40.71 | median | FPT | cosine 0.999905 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/thor/pi05.md) |
| B002 | FlashRT | π0.5 | Thor | NVFP4 + FA4 | 20.33 | median | FPT | cosine 0.999707；非全链路纯 FP4 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/thor/pi05.md) |
| B003 | FlashRT | π0.5 | RTX 5090 | FP8 skinny | 13.1 | p50 | FP5 | p95 13.18 ms | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) |
| B004 | FlashRT | π0.5 | RTX 5090 | NVFP4 prefix + decoder | 8.75 | p50 | FP5 | p95 8.78 ms；Gemma attention 保留 FP8 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md) |
| B005 | FlashRT | GR00T N1.7 | Thor | FP8 backbone + NVFP4 action head | 27.89 | median | FGT | 同表 model 24.26、preprocess 3.25 ms；不要将分项中位数相加 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) |
| B012 | EmbodiInfer | π0.5 | RTX 4090 | BF16 native optimized | 38.89 | mean | EP | forward 36.54；数值 gate 10/10 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B013 | vla.cpp | π0.5 | RTX 4090 | native BF16/F32 weights + F32 activations | 79.36 | mean | EP | EmbodiInfer 发布的竞品测试 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B014 | Embodied.cpp | π0.5 | RTX 4090 | native BF16/F32 weights + F32 activations | 89.99 | mean | EP | EmbodiInfer 发布；使用 correctness patch | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B015 | PhyAI | π0.5 | RTX 4090 | BF16 | 41.31 | mean | EP | forward 39.14；数值 gate 9/10，allow_numerical_mismatch | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B016 | EmbodiInfer | GR00T N1.7 | RTX 4090 | BF16 native optimized | 31.3 | mean | EG | forward 24.17；数值 gate 10/10 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| B017 | vla.cpp | GR00T N1.7 | RTX 4090 | BF16 GGUF + F32 activations | 38.7 | mean | EG | EmbodiInfer 发布的竞品测试 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| B018 | Embodied.cpp | GR00T N1.7 | RTX 4090 | BF16/F32 native | 73.17 | mean | EG | EmbodiInfer 发布；FlashAttention + Graph | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| B019 | PhyAI | GR00T N1.7 | RTX 4090 | BF16 | 34.48 | mean | EG | forward 27.18；数值 gate 9/10，allow_numerical_mismatch | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| B020 | EmbodiInfer | GR00T N1.7 | Thor | BF16 | 92.47 | mean | EG | 2026-09-08 版本；p50 90.14、p95 97.87、p99 109.21；forward 80.50 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| B021 | EmbodiInfer | π0.5 | Thor | BF16 historical | 164.05 | mean | EP | 2026-09-08 旧版本；p50 163.95、p95 165.01、p99 165.50；forward 160.90 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B022 | EmbodiInfer | π0.5 | RTX 4090 | BF16 | 39.77 | mean | EQ | 量化专项版本；forward 36.74 ms；显存 10.8 GiB | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) |
| B023 | EmbodiInfer | π0.5 | RTX 4090 | FP8 Triton | 121.76 | mean | EQ | 量化专项版本；forward 118.57 ms；显存 5.89 GiB | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) |
| B024 | EmbodiInfer | π0.5 | Thor | BF16 | 98.23 | mean | EQ | 量化专项版本；forward 94.84 ms；显存 10.8 GiB | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) |
| B025 | EmbodiInfer | π0.5 | Thor | FP8 W8A8 tensorwise | 251.89 | mean | EQ | 量化专项版本；forward 248.19 ms；显存 5.98 GiB | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) |
| B026 | EmbodiInfer | π0.5 | Thor | NVFP4 | 386.36 | mean | EQ | 量化专项版本；forward 382.66 ms；显存 5.46 GiB | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) |
| B052 | Isaac-GR00T + TensorRT | GR00T N1.7 | Thor | eager BF16 | 111.5 | median | NV | action cosine ≥0.99；不是任务成功率 | [原文](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) |
| B053 | Isaac-GR00T + TensorRT | GR00T N1.7 | Thor | torch.compile BF16 | 100.5 | median | NV | action cosine ≥0.99；不是任务成功率 | [原文](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) |
| B054 | Isaac-GR00T + TensorRT | GR00T N1.7 | Thor | TensorRT BF16 | 77.6 | median | NV | action cosine ≥0.99；不是任务成功率 | [原文](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) |
| B055 | Isaac-GR00T + TensorRT | GR00T N1.7 | Thor | TensorRT NVFP4 mixed | 35.9 | median | NV | action cosine ≥0.99；不是任务成功率 | [原文](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) |

## 模型、CUDA Graph 与不完整边界

| ID | 引擎 | 模型 | GPU | 数值格式 / 路径 | 延迟 ms | 统计 | 输入 profile | 备注 | 来源 |
|---|---|---|---|---|---:|---|---|---|---|
| B006 | FlashRT | GR00T N1.7 | Thor | FP8/NVFP4 | 23.23 | median | FGT | README native model 数值；排除 fresh CPU preprocessing 和物理动作解码 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/thor/groot-n17.md) |
| B007 | FlashRT | GR00T N1.7 | RTX 5090 | FP8 | 14.01 | GPU event median | FG5 | action cosine mean/min 0.995/0.979 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md) |
| B008 | FlashRT | GR00T N1.7 | RTX 5090 | NVFP4 mixed | 8.64 | GPU event median | FG5 | action cosine mean/min 0.988/0.967 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md) |
| B009 | FlashRT | GR00T N1.7 | RTX 4090 | FP8 eager | 32.6 | p50 | FG4 | mean 33.50；不是完整 RGB→物理动作 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) |
| B010 | FlashRT | GR00T N1.7 | RTX 4090 | FP8 graph hot | 9.98 | 2 replays mean | FG4 | 样本极少 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) |
| B011 | FlashRT | GR00T N1.7 | RTX 4090 | FP8 fused graph | 9.69 | 4 replays mean | FG4 | 样本极少 | [原文](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md) |
| B027 | EmbodiInfer | π0.5 | Thor | optimized BF16 | 70.189 | p50 | EN | 2026-10-10 prepared-input profile | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B028 | EmbodiInfer | π0.5 | Thor | FP8 action + NVFP4 prefix mixed | 58.054 | p50 | EN | 与完整 E2E 98.23 ms 不是同口径 | [原文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| B029 | vla.cpp | π0.5 | RTX 5090 | 表未固定 dtype | 29.2 | min of 5 rounds | VP | — | [原文](https://github.com/VinRobotics/vla.cpp/blob/main/docs/benchmark/rtx-5090.md) |
| B030 | vla.cpp | GR00T N1.7 | RTX 5090 | 表未固定 dtype | 19.5 | min of 5 rounds | VG | — | [原文](https://github.com/VinRobotics/vla.cpp/blob/main/docs/benchmark/rtx-5090.md) |
| B035 | Realtime-VLA | π0.5 | RTX 4090 | BF16 | 22.1 | median（脚本） | RT | 1 视角； | [原文](https://github.com/Dexmal/realtime-vla) |
| B036 | Realtime-VLA | π0.5 | RTX 4090 | BF16 | 29.2 | median（脚本） | RT | 2 视角； | [原文](https://github.com/Dexmal/realtime-vla) |
| B037 | Realtime-VLA | π0.5 | RTX 4090 | BF16 | 38.9 | median（脚本） | RT | 3 视角； | [原文](https://github.com/Dexmal/realtime-vla) |
| B038 | Realtime-VLA | π0.5 | RTX 5090 | BF16 | 20.1 | median（脚本） | RT | 1 视角；原表 | [原文](https://github.com/Dexmal/realtime-vla) |
| B039 | Realtime-VLA | π0.5 | RTX 5090 | BF16 | 26.6 | median（脚本） | RT | 2 视角；原表 | [原文](https://github.com/Dexmal/realtime-vla) |
| B040 | Realtime-VLA | π0.5 | RTX 5090 | BF16 | 34.2 | median（脚本） | RT | 3 视角；原表 | [原文](https://github.com/Dexmal/realtime-vla) |
| B041 | Realtime-VLA | π0.5 | RTX 5090 | BF16 | 21.3 | median（脚本） | RT | 2 视角；2026-09 优化表 | [原文](https://github.com/Dexmal/realtime-vla) |
| B042 | Realtime-VLA | π0.5 | RTX 5090 | BF16 | 27.4 | median（脚本） | RT | 3 视角；2026-09 优化表 | [原文](https://github.com/Dexmal/realtime-vla) |
| B043 | FlashVLA | π0.5 | RTX 4090 | BF16 | 26.7 | mean | FV | 2 视角；单步流式方法，不作为原版 10 步的直接速度排名 | [原文](https://github.com/z-lab/flashvla) |
| B044 | FlashVLA | π0.5 | RTX 4090 | BF16 | 36.8 | mean | FV | 3 视角；单步流式方法，不作为原版 10 步的直接速度排名 | [原文](https://github.com/z-lab/flashvla) |
| B045 | FlashVLA | π0.5 | RTX 5090 | BF16 | 20.3 | mean | FV | 2 视角；单步流式方法，不作为原版 10 步的直接速度排名 | [原文](https://github.com/z-lab/flashvla) |
| B046 | FlashVLA | π0.5 | RTX 5090 | BF16 | 27.1 | mean | FV | 3 视角；单步流式方法，不作为原版 10 步的直接速度排名 | [原文](https://github.com/z-lab/flashvla) |
| B064 | ApxInf / APXinf-robo | π0.5 | Thor | BF16 | 72.45 | p50 | AP | — | [原文](https://github.com/RLinf/APXinf-robo) |
| B065 | ApxInf / APXinf-robo | π0.5 | Thor | FP8 | 41.16 | p50 | AP | — | [原文](https://github.com/RLinf/APXinf-robo) |
| B066 | ApxInf / APXinf-robo | π0.5 | RTX 4090 | BF16 | 31.38 | p50 | AP | — | [原文](https://github.com/RLinf/APXinf-robo) |
| B067 | ApxInf / APXinf-robo | π0.5 | RTX 4090 | INT8 | 25.99 | p50 | AP | — | [原文](https://github.com/RLinf/APXinf-robo) |
| B068 | ApxInf / APXinf-robo | π0.5 | Thor | BF16 | 44.05 | p50 | AP | STEP warm-start 单步变体；将 10 步改为 1 步，不可直接等同比较 | [原文](https://github.com/RLinf/APXinf-robo) |
| B069 | ApxInf / APXinf-robo | π0.5 | Thor | FP8 | 26.32 | p50 | AP | STEP warm-start 单步变体；将 10 步改为 1 步，不可直接等同比较 | [原文](https://github.com/RLinf/APXinf-robo) |
| B070 | ApxInf / APXinf-robo | π0.5 | RTX 4090 | BF16 | 20.36 | p50 | AP | STEP warm-start 单步变体；将 10 步改为 1 步，不可直接等同比较 | [原文](https://github.com/RLinf/APXinf-robo) |
| B071 | ApxInf / APXinf-robo | GR00T N1.7 | Thor | BF16 | 51.834 | best recorded p50 | AG | 1 视角；独立闭环评测 Thor BF16 94/100、FP8 92/100，非延迟 fixture 的任务成功率 | [原文](https://github.com/RLinf/APXinf-robo/blob/main/doc/gr00t-n1.7.md) |
| B072 | ApxInf / APXinf-robo | GR00T N1.7 | Thor | BF16 | 54.216 | best recorded p50 | AG | 2 视角；独立闭环评测 Thor BF16 94/100、FP8 92/100，非延迟 fixture 的任务成功率 | [原文](https://github.com/RLinf/APXinf-robo/blob/main/doc/gr00t-n1.7.md) |
| B073 | ApxInf / APXinf-robo | GR00T N1.7 | Thor | FP8 | 32.557 | best recorded p50 | AG | 1 视角；独立闭环评测 Thor BF16 94/100、FP8 92/100，非延迟 fixture 的任务成功率 | [原文](https://github.com/RLinf/APXinf-robo/blob/main/doc/gr00t-n1.7.md) |
| B074 | ApxInf / APXinf-robo | GR00T N1.7 | Thor | FP8 | 35.436 | best recorded p50 | AG | 2 视角；独立闭环评测 Thor BF16 94/100、FP8 92/100，非延迟 fixture 的任务成功率 | [原文](https://github.com/RLinf/APXinf-robo/blob/main/doc/gr00t-n1.7.md) |

## 边界未充分确认或历史参考

| ID | 引擎 | 模型 | GPU | 数值格式 / 路径 | 延迟 ms | 统计 | 输入 profile | 备注 | 来源 |
|---|---|---|---|---|---:|---|---|---|---|
| B031 | PhyAI | π0.5 | Thor | BF16（作者博客说明） | 107.7 | 图未说明统计量 | PH | — | [原文](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg) |
| B032 | PhyAI | π0.5 | RTX 5090 | BF16（作者博客说明） | 28.6 | 图未说明统计量 | PH | — | [原文](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg) |
| B033 | PhyAI | GR00T N1.7 | Thor | BF16（作者博客说明） | 95.4 | 图未说明统计量 | PH | — | [原文](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg) |
| B034 | PhyAI | GR00T N1.7 | RTX 5090 | BF16（作者博客说明） | 20.6 | 图未说明统计量 | PH | — | [原文](https://github.com/mingti-org/phyai/blob/main/assets/performance-comparison.svg) |
| B047 | EagleVLA-Edge | π0.5 | Thor | 表未确认；示例 FP16 | 457.9 | 阶段 p50 之和 | EA | naive | [原文](https://github.com/PKU-SEC-Lab/EagleVLA-Edge) |
| B048 | EagleVLA-Edge | π0.5 | Thor | 表未确认；示例 FP16 | 240 | 阶段 p50 之和 | EA | CUDA Graph | [原文](https://github.com/PKU-SEC-Lab/EagleVLA-Edge) |
| B049 | EagleVLA-Edge | π0.5 | Thor | 表未确认；示例 FP16 | 202.8 | 阶段 p50 之和 | EA | Graph + buffer + GPU KV + unroll | [原文](https://github.com/PKU-SEC-Lab/EagleVLA-Edge) |
| B050 | InstinctFlash | π0.5 | Thor | FP8 historical | 51.85 | p50 | IN | 旧引擎；不能代表当前实现 | [原文](https://github.com/General-Instinct/InstinctFlash) |
| B051 | InstinctFlash | GR00T N1.7 | Thor | BITEXACT（dtype 未完整声明） | 117.3 | p50 | IN | — | [原文](https://github.com/General-Instinct/InstinctFlash) |
| B056 | OpenPI + TensorRT（Jetson AI Lab） | π0.5 | Thor | torch.compile BF16 | 132 | 约值；统计未明确 | JP | 发布方 total；model 约 128 ms | [原文](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) |
| B057 | OpenPI + TensorRT（Jetson AI Lab） | π0.5 | Thor | TensorRT FP8 | 54 | 约值；统计未明确 | JP | 发布方 total；model 约 53 ms | [原文](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) |
| B058 | OpenPI + TensorRT（Jetson AI Lab） | π0.5 | Thor | TensorRT FP8 + NVFP4 | 49 | 约值；统计未明确 | JP | 发布方 total；model 约 48 ms | [原文](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) |
| B059 | Isaac-GR00T（Jetson AI Lab 教程） | GR00T N1.7 | Thor | eager | 126 | 约 median | JG | 教程称 E2E；旧版/单相机，不和最新官方双相机完整调用混排 | [原文](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) |
| B060 | Isaac-GR00T（Jetson AI Lab 教程） | GR00T N1.7 | Thor | torch.compile | 105 | 约 median | JG | 教程称 E2E；旧版/单相机，不和最新官方双相机完整调用混排 | [原文](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) |
| B061 | Isaac-GR00T（Jetson AI Lab 教程） | GR00T N1.7 | Thor | TensorRT BF16 | 81 | 约 median | JG | 教程称 E2E；旧版/单相机，不和最新官方双相机完整调用混排 | [原文](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) |
| B062 | Isaac-GR00T（Jetson AI Lab 教程） | GR00T N1.7 | Thor | TensorRT FP8 | 44 | 约 median | JG | 教程称 E2E；旧版/单相机，不和最新官方双相机完整调用混排 | [原文](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) |
| B063 | Isaac-GR00T（Jetson AI Lab 教程） | GR00T N1.7 | Thor | TensorRT NVFP4 mixed | 40 | 约 median | JG | 教程称 E2E；旧版/单相机，不和最新官方双相机完整调用混排 | [原文](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) |

## 输入、计时与复现条件

逐行 profile 是记录的一部分。未知值并不表示引擎没有该配置。

### FPT

B=1；2 路 RGB；pi05_libero；10 去噪步；H=10，输出 10×7；8 个真实 LIBERO observation；200 warmup / 1000 次；固定 prompt 在 setup 阶段；Thor JP7.2/CUDA13、MAXN、锁频。

### FP5

B=1；2 路 RGB；pi05_libero，无 state-in-prompt；10 去噪步；H=10，输出 10×7；8 个真实 LIBERO observation；30 warmup / 200 次同步完整调用；2026-10-16 更新。

### FGT

B=1；单相机真实 fixture；GR00T-N1.7-LIBERO/libero_10；4 去噪步；内部 40×132，物理输出 16×7；Thor JP7.2 MAXN 锁频；指南协议 20 warmup / 100 次，release validator 数值另见比较文档。

### FG5

B=1；2 视角；4 去噪步；GPU events 200 次，fresh backbone graph + action graph；分辨率/物理解码未在引用表中明确；非完整 CPU 输入到物理动作。

### FG4

真实 2-view fixture；H=40，内部输出 1×40×132；FP8 layout nk；2026-07-02/03；graph 数值仅来自 2 次或 4 次 replay，非大量请求分布；边界不含完整前后处理。

### EP

B=1；2×224² RGB + masked 空相机，state + instruction；10 去噪步；内部 50×32，物理输出 50×7；pi05_libero_finetuned_v044_gitcode；1600 个 LIBERO-10 observation（10任务×10demo×16帧），10 次跨任务 warmup；含预处理、噪声、H2D/D2H、同步、模型、动作反归一化。

### EG

B=1；2×256² RGB；GR00T-N1.7-LIBERO/libero_10 revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21；4 去噪步；内部 40×132，物理输出 16×7；1600 个真实 LIBERO-10 observation、10 warmup；原版图像翻转/裁剪、Qwen 模板及 percentile state normalization；完整 CPU 输入到物理动作。

### EQ

π0.5；B=1；1600 个真实 LIBERO observation；2×224² + masked 空相机；10 步，H=50；独立量化实验，不能将差异全部归因于 dtype；BF16 使用 native/Inductor/Graph，FP8/NVFP4 使用不同低精度后端；各目录提供配置。

### VP

B=1；合成输入；2×224²，lang=16，10 步，H=50，action_dim=7；in-process predict，5 轮最小值；权重 as shipped，表未明确统一 dtype；PR32 维护者引用数据未重新测量。

### VG

B=1；合成输入；1×256²，lang=16，4 步，H=40，action_dim=132；in-process predict，5 轮最小值；未计完整机器人预处理/物理动作返回。

### PH

README 性能图 single-request latency；未充分披露端到端边界、视角数、H、步数及采样协议；作者博客说明 BF16 非量化，不能将另一张 batch 实验的形状套用到本图。

### RT

B=1 的合成 forward benchmark；GPU 预置 BF16 224² 图像、语言 embedding、噪声；源码默认 H=50、prompt_len=0、3 warmup / 100 次同步计时中位数；README 历史表没有完整命令，默认值不作为历史配置已核实；不含 CPU 预处理/传输。

### FV

B=1；224²；BF16；专用训练配方，rolling buffer 单步流式生成，不能等同原模型 10 步；配置 buffer=5、chunk=10；10 warmup / 100 次均值；源码在计时前预处理和传输输入；历史表实际权重未固定。

### EA

Thor；2×224²；10 去噪步、10 actions；分阶段 p50 求和，不是完整请求样本的 p50；quickstart FP16，表未独立确认 precision。

### IN

Thor；README 历史结果 2026-09-15；predict p50，完整边界/输入形状未充分披露；复现脚本当前示例 5 warmup / 20 次，不能直接视为历史测试协议；π0.5 历史 FP8 引擎已被新实现替代且未重新测量。

### NV

B=1；GR00T-N1.7-LIBERO/libero_10；2×256²，156 VLM tokens；完整 32 DiT blocks；H=40、4 去噪步；episode0/step0 固定 fixture；Thor 120W 锁频、CPU8/inter1；10 warmup，3×20 次合并中位数，35.9 ms 优化项为独立 20 次；含预处理及物理动作解码，不含 RPC。

### JP

Thor JP7.2 MAXN；pi05_libero；H=10；教程及比较文档给出 10 去噪步；相机数、统一 batch/采样统计未在性能表完整披露；OpenPI 固定提交 + 外部部署脚本；total 与 model 分列，精确完整物理动作边界待确认。

### JG

Thor JP7.2 MAXN；B=1，单相机、4 去噪步；libero_10；5 warmup / 20 次中位数；固定旧版 Isaac-GR00T + patch；教程称 E2E，但物理动作解码边界未与最新版官方完整调用对齐。

### AP

B=1；2×224² NHWC uint8；10 flow steps，H=10；steady-state CUDA Graph replay P50；README 随机权重示例 token_count=10、10 warmup / 100 或默认 30 次，表未固定实际 checkpoint，不能将延迟表直接当作已验证任务精度的同次运行。

### AG

B=1；1 或 2 视角固定 processor tensors→returned actions；best recorded model-core P50，排除原始图像 processor；官方 libero_10 checkpoint revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21，episode0/step0、零 diffusion noise；示例 10 warmup / 50 次 CUDA Graph；引用页面未完整列分辨率、H、去噪步数。

### EN

B=1；π0.5 RLinf-Pi05-LIBERO-SFT；2×224² + masked 相机；H=10、10 步；6 个真实 observation，60 次同步 wall P50；prepared-input/model 优化 profile，不确认完整原始 CPU observation 边界；matmul=high。

## 数值正确性、任务精度与版本限制

1. **EmbodiInfer 的 RTX 4090 对照是本轮最接近同输入跨引擎比较的数据**，但发布方是其中一个引擎作者，不是独立中立复测。π0.5 的 vla.cpp 提交为 `f386c16347094eb2cb183edf39677cf1b6bcc750`，Embodied.cpp 为 `1dad33f2c87ee1d390808cb5d776cd8c998f4a36` 且加入 correctness patch，PhyAI 为 `a0abb211c4b05b21b766d0f6f6840eed72a8fcec`。PhyAI 两个模型均为数值 gate 9/10、允许 mismatch；这不等于 90% 任务成功率。原文称原始 JSON 为本地归档，本轮未取得这些原始测试输出。[π0.5 报告](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md)、[GR00T 报告](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md)。
2. **FlashRT π0.5 的 RTX 5090 更新已明确完整调用边界**。使用 2026-10-16 的 13.10/8.75 ms，不把旧的 state-in-prompt 或旧 loader 数字作为当前主结果。GR00T RTX 5090 的 8.64 ms 仍然只是 GPU 时间。[π0.5](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_prefix_nvfp4.md)、[GR00T](https://github.com/flashrt-project/FlashRT/blob/main/docs/groot_n17_rtx_nvfp4.md)。
3. **FlashRT RTX 4090 部署文档中的 π0 30–52 ms 是估算且模型不同**，不能当作 π0.5 实测，因此未录入。N1.6 数字也不移作 N1.7。N1.7 graph 9.69 ms 只有少量 replay，不代表大量真实请求的 p50/p95。[比较说明](https://github.com/flashrt-project/FlashRT/blob/main/docs/benchmark_comparison.md)。
4. **FlashRT 的 Thor GR00T 23.23、27.89 与教程约 40 ms 不是同一计时边界/版本**。当前完整 CPU RGB→物理动作使用 27.89 ms；23.23 保留在 model-only。Jetson AI Lab 的旧单相机教程与最新官方双相机部署表单列。[FlashRT](https://github.com/flashrt-project/FlashRT/blob/main/docs/thor/groot-n17.md)、[官方部署](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)、[旧教程](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)。
5. **EmbodiInfer 低精度专项不一定更快**：Thor FP8 251.89、NVFP4 386.36 ms 均慢于该实验 BF16 98.23 ms；后端及 profile 差异很大，不能得出“FP8/NVFP4 本身一定更慢”的结论。2026-10-10 的 58.054 ms 属另一个 prepared-input 优化 profile，不能替换完整 E2E。[量化报告](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md)、[profile 更新](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md)。
6. **FlashVLA、ApxInf STEP 单步模式改变去噪算法**，和原始 10 步 π0.5 分开看，需要相同任务上的闭环精度验证。ApxInf 独立公布 π0.5 LIBERO-10、500 episodes、H10/replan5/seed7：Thor BF16 92.8%、FP8 92.2%，reference 92.4%；这些数字不能自动绑定到 README 随机权重延迟或所有单步路径。[ApxInf README](https://github.com/RLinf/APXinf-robo)。
7. **ApxInf GR00T 闭环精度**：Thor BF16 94/100，FP8 92/100；均为 LIBERO-10 两视角、每任务 10 episodes，样本有限；延迟则来自固定 processor tensor model-core fixture。[GR00T 说明](https://github.com/RLinf/APXinf-robo/blob/main/doc/gr00t-n1.7.md)。
8. cosine 只衡量同输入动作接近程度；例如 FlashRT 的低精度 cosine 和 NVIDIA 的 ≥0.99 都不是闭环任务成功率。未公开成功率的测试不补填。

## 覆盖缺口与下一轮实测优先项

- FlashRT π0.5 / RTX 4090：未找到可确认为该模型的完整 E2E 实测。GR00T / RTX 5090：有模型数字，仍需完整 CPU RGB→物理动作。
- Realtime-VLA、FlashVLA：本轮只取得 4090/5090 π0.5 数据。没有把其 π0 或其它 GPU 数字混入。
- InstinctFlash、EagleVLA-Edge：取得 Thor 参考数据；未找到 4090/5090 的同模型可信完整表。EagleVLA-Edge 的 N1.7 模型路径不等于已有目标性能结果。
- ApxInf：取得 Thor 两模型及 4090 π0.5；未取得 5090 结果或 4090 N1.7 对齐性能。
- 通用 LLM 引擎与缺少目标数字的项目仍见引擎清单，不能将空白当作“不支持”。
- 下一轮建议先固定：π0.5 两相机 224²、10 步、分别 H10/H50；N1.7 两相机 256²、4 步、内部 H40/物理输出16×7；同 checkpoint 与 normalization、同 prompt/state、同真实 observation 序列。计时同时记录 CPU observation→CPU physical action 与 GPU model，至少 mean/p50/p95/p99，并固定 warmup、频率、功耗、软件版本及闭环任务成功率。

## 来源与追溯说明

主要依据各仓库 README、部署指南、benchmark 文档；当 README 缺少边界时补查 benchmark 源码，而非凭名称推断。

- Realtime-VLA 计时边界：[benchmark.py](https://github.com/Dexmal/realtime-vla/blob/main/benchmark.py)。
- FlashVLA 计时边界：[benchmark_latency.py](https://github.com/z-lab/flashvla/blob/main/benchmarks/benchmark_latency.py)、[配置](https://github.com/z-lab/flashvla/blob/main/benchmarks/configs/latency_flashvla.yaml)。
- PhyAI 数值图的 BF16 背景：[作者技术文章](https://mingti-org.github.io/phyai-blog/blogs/260718-phyai/)。图中未披露的输入条件仍保留未知。
- InstinctFlash 历史结果与复现：[README](https://github.com/General-Instinct/InstinctFlash)、[REPRODUCE.rst](https://github.com/General-Instinct/InstinctFlash/blob/main/REPRODUCE.rst)。

结构化记录保存在 `research_data.json`，ID 与本文件对应；profile 与 source 通过键关联。

