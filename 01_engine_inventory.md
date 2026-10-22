# 开源推理引擎并集与模型支持

调研日期：2026-10-22。目标：单卡、batch=1、低延迟；硬件优先 Thor、RTX 4090、RTX 5090。

本文件收录 21 项：前述 18 个引擎、OpenPI 参考实现，以及本轮新增的 ApxInf 和 Jetson AI Lab OpenPI+TensorRT 部署路线。部署路线不一定是独立引擎，分类见最后一列。GR00T “1.7N”按官方名称 **N1.7** 处理。

“支持”表示仓库存在相应模型实现或配方，不表示已验证所有 checkpoint、所有硬件或端到端性能。“未找到/未确认”仅表示本轮公开材料证据不足，不等于不支持。包含 Qwen 骨干、Qwen-Drive 或 Qwen R2R 不自动等于支持通用 Qwen LLM 服务。通用 LLM 引擎保留在并集中，本轮不扩展 Qwen 性能测试。

| 引擎 / 来源 | 一句话描述 | π0.5 | GR00T | Qwen | README/官方文档披露的技术 | 分类 |
|---|---|---|---|---|---|---|
| [FlashRT](https://github.com/flashrt-project/FlashRT) | 为具身模型提供专用 CUDA 内核和低精度推理路径。 | 支持 | 支持 N1.7 | 支持，具体变体见仓库 | CUDA Graph、融合内核、FP8/NVFP4、前缀复用 | 原有 |
| [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) | 以完整策略推理及可比较的输入输出协议优化具身模型。 | 支持 | 支持 N1.7 | Qwen R2R 路线；不等同通用 Qwen 服务 | Inductor、Triton、CUDA Graph、前缀 KV 复用、量化 | 原有 |
| [PhyAI](https://github.com/mingti-org/phyai) | 面向具身策略的 Python 推理优化框架。 | 支持 | 支持 N1.7 | 支持 Qwen3.5 / VL 等 | 算子融合、CUDA Graph、autotune、KV/时间步缓存 | 原有 |
| [Realtime-VLA](https://github.com/Dexmal/realtime-vla) | 面向 π 系列单请求延迟的专用推理实现。 | 支持 | 未找到 | 未找到 | 融合算子、编译、静态图执行 | 原有 |
| [InstinctFlash](https://github.com/General-Instinct/InstinctFlash) | 面向 Jetson 的具身模型部署和加速实现。 | 支持 | 支持 N1.7 | 未找到 | FP8、专用部署 profile、数值一致性验证 | 原有 |
| [vla.cpp](https://github.com/VinRobotics/vla.cpp) | 基于 GGML/GGUF 的轻量 C/C++ VLA 推理引擎。 | 支持 | 支持 N1.5 / N1.6 / N1.7 | 未确认独立 Qwen 推理接口 | GGUF、GGML CUDA、FlashAttention、轻量运行时 | 原有 |
| [EagleVLA-Edge（Jetson-PI-Edge）](https://github.com/PKU-SEC-Lab/EagleVLA-Edge) | 面向 Jetson 的 VLA 边缘推理优化工程。 | 支持 | 仓库提供 N1.7 路线 | 未确认独立 Qwen 推理接口 | CUDA Graph、预分配 buffer、GPU KV cache、去噪展开 | 原有 |
| [Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp) | 将具身模型部署到 GGML/GGUF C++ 运行时。 | 支持 | 上游支持 N1.7 | 未确认独立 Qwen 推理接口 | GGUF、CUDA、FlashAttention、CUDA Graph | 原有；纠正上游 |
| [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | 面向边缘设备的 TensorRT LLM/VLM 及实验性 VLA 部署栈。 | 实验性支持 | 未找到完整 N1.7 路线 | 支持，具体变体见支持矩阵 | TensorRT、量化、静态部署引擎 | 原有 |
| [Isaac-GR00T + TensorRT](https://github.com/NVIDIA/Isaac-GR00T) | NVIDIA 官方 GR00T 模型与 TensorRT 部署路线。 | 不属于此仓库模型范围 | 官方支持 N1.7 | 模型内部骨干不等于独立 Qwen 服务 | torch.compile、TensorRT、FP8/NVFP4 混合精度 | 原有 |
| [SGLang](https://github.com/sgl-project/sglang) | 通用 LLM/VLM 服务引擎，提供 OpenPI π0.5 推理配方。 | 支持 OpenPI 配方 | 未找到完整 N1.7 路线 | 支持 | RadixAttention、CUDA Graph、优化 attention、量化 | 原有 |
| [FlashVLA](https://github.com/z-lab/flashvla) | 通过流式动作生成及推理融合降低 VLA 延迟。 | 支持专用训练/推理配方 | 未找到 | 未找到 | rolling buffer、单步流式去噪、torch.compile、投影融合 | 原有 |
| [Robot.cpp](https://github.com/Robot-cpp/robot.cpp) | 基于 llama.cpp 的跨平台机器人模型服务框架。 | README 明确 π0；未确认 π0.5 | 未找到 | StarVLA/Qwen 相关路线；非等同通用 Qwen 引擎 | 模块化 GGUF、选择性量化、C++、TCP 服务 | 原有 |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | 面向本地低批量量化 LLM 的 GPU 推理引擎。 | 未找到完整 VLA | 未找到完整 VLA | 支持多个 Qwen 文本/VL 变体 | EXL3 量化、专用 GPU 内核、缓存优化 | 原有 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 基于 GGML/GGUF 的跨平台本地 LLM/VLM 推理引擎。 | 主仓未确认；VLA fork 另列 | 主仓未确认；VLA fork 另列 | 支持多个 Qwen 文本/VL 变体 | GGUF 量化、CPU/GPU offload、优化 attention | 原有 |
| [LMDeploy / TurboMind](https://github.com/InternLM/lmdeploy) | 提供 TurboMind/PyTorch 后端的 LLM/VLM 部署框架。 | 未找到完整 VLA | 未找到完整 VLA | 支持；依模型和后端而异 | 量化、KV cache、优化 attention、批处理调度 | 原有 |
| [vLLM](https://github.com/vllm-project/vllm) | 面向 LLM/VLM 的高吞吐服务和推理引擎。 | 未确认完整 π0.5 策略 | 未确认完整 N1.7 策略 | 支持 | PagedAttention、CUDA Graph、量化、连续批处理 | 原有 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 面向 GPU 的 LLM 优化推理库。 | 未确认完整 π0.5 策略 | 不能将 TensorRT 部署等同此库直接支持 | 支持 | TensorRT/优化内核、量化、KV cache、图执行 | 原有 |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | π 系列官方训练与推理参考实现，用作基线。 | 官方支持 | 未找到 | 未找到 | JAX/JIT、PyTorch/compile、动作分块 | 参考实现 |
| [OpenPI + TensorRT（Jetson AI Lab）](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) | 将 OpenPI π0.5 拆分并部署到 Thor 的 TensorRT 实验路线。 | 支持教程配方 | 非此教程范围 | 非此教程范围 | TensorRT、FP8、NVFP4、模型拆分 | 本轮补充部署路线 |
| [ApxInf / APXinf-robo](https://github.com/RLinf/APXinf-robo) | 以 Rust/CUDA 推理核心和 Python 机器人接口部署 VLA。 | 支持 | 支持 N1.7 | Qwen-Drive；不等同通用 Qwen LLM | Rust、专用 CUDA、CUDA Graph、FP8/INT8、可选 STEP | 本轮新增 |

## 去重与修正

- LMDeploy/TurboMind 作为一项；EagleVLA-Edge/Jetson-PI-Edge 作为一项；ApxInf 核心及 APXinf-robo 接入层作为一项。
- Embodied.cpp 使用 [SEU-PAISys 上游](https://github.com/SEU-PAISys/Embodied.cpp)。此前提到的 [tiiuae/Embodied.cpp-edge](https://github.com/tiiuae/Embodied.cpp-edge) 是另一 fork，不能将其“GR00T 计划支持”状态套用到上游；上游已有 N1.7。
- TensorRT-Edge-LLM、TensorRT-LLM、Isaac-GR00T 的 TensorRT 部署、Jetson AI Lab 的 OpenPI+TensorRT 配方是不同项目/路线，不能互相替代支持证据。
- 使用低精度、CUDA Graph 或通用高吞吐引擎并不保证 batch=1 延迟更低。需要核对完整输入输出和硬件实测。

## 本轮未找到目标硬件/模型组合数值的项目

TensorRT-Edge-LLM、SGLang、Robot.cpp、ExLlamaV3、llama.cpp、LMDeploy/TurboMind、vLLM、TensorRT-LLM 未取得可用于本轮 π0.5/N1.7 × Thor/4090/5090 的独立、明确公开延迟。SGLang π0.5 文档有 H100/H200 结果，但不移植为目标 GPU 结果。OpenPI 作为基线的数字出现在部署路线或他方对比中，不另外伪造独立实测。

更具体的 π0.5 配方：[SGLang](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/vla/OpenPI/Pi0.5.mdx)、[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md)。

本表是材料调研，未安装、编译或实机验证这些引擎；技术栏目是项目披露，不代表每个模型路径都启用全部技术。

