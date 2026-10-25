# RTX 5090 × GR00T N1.7：新增已实现 Recipe 测试

本轮遵循用户限定：沿用 FlashRT 已测的三视角实验；已有有效结果不重跑；优先新框架；只适配上游已实现的路径，不补写引擎量化、CUDA kernel 或模型计算。有效 Run 表示完成协议检查和计时，**不表示通过任务精度或数值验收门槛**。

## 固定条件

- 官方 base checkpoint revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`，仍使用服务器原 Hugging Face 权重。
- 原冻结评估集 `datasets/groot-n17-v3`，manifest SHA256 `562a86fc238ca10adcc2b623be09bf796b74f1403fbdd5ab6f6d1785335c2297`。
- B1、三视角 256×256、4 步去噪、完整 H40×A132 normalized action；相同 state、文本和初始 noise。
- prepared GPU tensor（learned embeddings 之前）到完整 GPU action 的同步 wall latency；含内部拷贝与 CPU 调度，不含外部预处理、外部 H2D/D2H、构建及校准。
- 8 个 CPU threads，10 次预热、100 次计时，GPU lock 串行执行；沿用 A/B/A、输入敏感性、输入未修改、输出 shape/finite 检查。
- 不接模拟器，阈值仍为 null；小型合成 fixture 结果不能解释成任务成功率。

新量化路径要求多条校准样本，因此新增独立的 `datasets/groot-n17-calibration-v2`：原 calibration 样本加 7 条独立 seed 的同格式样本，共 8 条；不使用原 evaluation 样本。manifest SHA256 为 `73de684de0eaeaf679ac840f9c2379738526d65718216d7f906de371cb708db3`。原评估集未变。历史 FlashRT 的校准条件与本轮新 PTQ 路径不同，应保留这项区别。

## 范围与排除理由

| 引擎 | 本轮处理 |
|---|---|
| FlashRT | 跳过已测 FP8/NVFP4；尝试已有 FP8 backbone + FP16 action head。该路径的图状态检查需适配为逐步图；图像响应失败时不计入有效结果，不修改其缓存计算。 |
| EmbodiInfer | BF16 优化路径已有结果，跳过。FP8/NVFP4/INT8 候选需要给 GR00T 新增量化接线，依照本轮要求排除。 |
| PhyAI | 跳过 BF16；通过已有 Kernel Policy 配置 LLM 的 FP8、NVFP4、INT8、INT4 路径。DiT 空前缀所需的引擎修改不做。保存实际量化层清单；动态 A8 的图模式若失败，再测试现成 eager 开关。 |
| NVIDIA PyTorch/TensorRT | 新接官方现成 full-pipeline baseline、optimized、FP8、mixed-NVFP4，以及官方 DiT torch.compile 配方。 |
| FoldQuantVLA/TensorRT | 新接已有 W8A8、AWQ W4A16、SmoothQuant W8A8 配方；使用原生插件，不将 fake-quant PyTorch 执行当作低比特推理结果。 |
| vla.cpp | 现有公开 `vla_predict` 使用 host 图像/状态/noise，内部进行视觉准备并返回 host actions；`precomputed_img_emb` 会绕过视觉模型，不符合本协议。缺少当前协议所需的 prepared GPU 输入和 GPU action 接口，本轮不为其修改推理实现。 |
| Embodied.cpp | 现有 `vla::predict` 使用 host `ImageView`/state/noise 并返回 `std::vector<float>`，GR00T 路径包含内部视觉预处理。现成 EmbodiInfer benchmark bridge 同样使用这个 host API；本轮不将它的耗时混入 GPU tensor 协议。 |

C++ 接口核查对应 [固定版本源码证据及完整候选清单](../../recipes/groot-n17-5090-candidates.md)。实验性、需新增实现、以及改变 workload 的候选不因出现在待测表中就自动纳入本轮。

## 结果与证据

实验进行中；最终结果由 [run-index.json](run-index.json) 汇总。`runs/` 保留 summary、逐请求延迟和失败诊断；`log-tails/` 是服务端日志末尾摘录，完整日志位置及 SHA256 见 `server-log-manifest.json`。中间失败重试也保留，不用后一次结果覆盖前一次。

`assets/` 保存导出输入绑定、engine hash 和已退休中间产物 hash。为节约服务器磁盘，成功测量后可移除可重建的 ONNX/量化中间目录；保留 engine、源权重、冻结输入及重建脚本。

## 复现入口

实际执行的完整命令保存在各 Run 的 `stages.json` 或 `batch.json`。运行单个已构建 Recipe：

```bash
cd /data/inference-baseline
git pull --ff-only
bash scripts/run_5090.sh groot-trt-optimized artifacts/NEW_UNIQUE_RUN_ID
# 或 phyai-llm-nvfp4-a16 / foldquant-w8a8 等配置名。
```

从校准/导出开始，使用 `scripts/run_deployment_recipe.py`，并在外层持有 `artifacts/gpu-0.lock`；FoldQuant 从 `--start quantize` 开始，NVIDIA 从默认 export 开始。三视角校准集由 `scripts/prepare_groot_calibration.py` 生成至新目录，不覆盖原 fixture。

NVIDIA 使用 `.venv-groot-trt`，FoldQuant 使用 `.venv-foldquant`；从已有 PyTorch 2.9.0+cu128 / Transformers 4.57.3 / TensorRT 10.15.1.29 环境复用依赖文件建立独立 venv，再安装各自补充依赖。FoldQuant 初始依赖采用文件硬链接，安装器更新只替换其环境内的文件。量化部署按 ModelOpt 0.39.0 的约束使用 ONNX 1.19.1；未量化 optimized 的首个成功 Run 构建时使用 ONNX 1.20.1。各 Run 的版本与环境锁优先于概述。

上游源码均位于仓库 `third_party/`，通过固定提交及 `python_paths` 选择；Adapter 负责数据边界、噪声绑定和运行状态检查。导出工具只替换数据集/Policy 的 I/O 包装，校准算法、量化、ONNX 图及 TensorRT 执行由上游实现提供。
