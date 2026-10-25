# RTX 5090 × GR00T N1.7：新增已实现 Recipe 测试

本轮遵循用户限定：沿用 FlashRT 已测的三视角实验；已有有效结果不重跑；优先新框架；只适配上游已实现的路径，不补写引擎量化、CUDA kernel 或模型计算。有效 Run 表示完成协议检查和计时，**不表示通过任务精度或数值验收门槛**。

本轮已完成：16 个主比较配方加 4 个 eager 诊断配方，共 20 个配置，得到 **12 条有效测量、8 条协议检查失败记录**。新增 NVIDIA TensorRT 与 FoldQuant 两个实现家族；历史成功路径未重跑。完整延迟、p95、误差与原始记录见 [实测表](measured.md)。

## 固定条件

- 官方 base checkpoint revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`，仍使用服务器原 Hugging Face 权重。
- 原冻结评估集 `datasets/groot-n17-v3`，manifest SHA256 `562a86fc238ca10adcc2b623be09bf796b74f1403fbdd5ab6f6d1785335c2297`。
- B1、三视角 256×256、4 步去噪、完整 H40×A132 normalized action；相同 state、文本和初始 noise。
- prepared GPU tensor（learned embeddings 之前）到完整 GPU action 的同步 wall latency；含内部拷贝与 CPU 调度，不含外部预处理、外部 H2D/D2H、构建及校准。
- 8 个 CPU threads，10 次预热、100 次计时，GPU lock 串行执行；沿用 A/B/A、输入敏感性、输入未修改、输出 shape/finite 检查。
- 不接模拟器，阈值仍为 null；小型合成 fixture 结果不能解释成任务成功率。

实际 prepared layout 为 `pixel_values=[768,1536]`、`image_grid_thw=[3,3]`、`input_ids/attention_mask=[1,217]`、`state=[1,1,132]`、`noise=[1,40,132]`，embodiment ID 为 24。PyTorch TF32 按 Runner 关闭；TensorRT 保留上游构建器策略，不额外改写其精度标志。FP32 网络/输入类型不应自行解释成所有底层乘法都强制 IEEE FP32。

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

有效延迟与误差见 [实测表](measured.md)，包括重试的全部记录由 [run-index.json](run-index.json) 汇总。`runs/` 保留 summary、逐请求延迟和失败诊断；`log-tails/` 是服务端日志末尾摘录，完整日志位置及 SHA256 见 `server-log-manifest.json`。中间失败重试也保留，不用后一次结果覆盖前一次。

本轮执行 16 个主比较 Recipe，另为 PhyAI 增加 4 个 eager 诊断配方。原候选表的 51 行逐一映射到 [覆盖状态](candidate-coverage.json)：既有成功结果跳过、需要补写引擎计算或不兼容 GPU 边界的候选排除、执行消融和实验性 W4A4/backbone 路径延后；不声称 51 条均已实测。

### 精度范围与执行方式

| 配置前缀 | 实际范围 |
|---|---|
| `groot-trt-baseline` | 官方 full-pipeline，ViT FP32 网络、LLM/DiT/辅助模块 BF16，baseline 执行 |
| `groot-trt-optimized` | 相同未量化范围，optimized runner、cross-K/V 请求内复用、去噪图 |
| `groot-trt-fp8` | 上游 FP8 full-pipeline 配方；保留其混合精度和辅助层策略 |
| `groot-trt-nvfp4` | ViT FP8；LLM NVFP4，o/down FP8；DiT FFN NVFP4、attention FP8；辅助 FP16 |
| `groot-trt-compile` | 原模型 BF16，官方 DiT `torch.compile(mode="max-autotune")`；其余为 PyTorch |
| `phyai-llm-*` | 仅 LLM 的 64 个融合投影层量化；vision/action head 仍为原生 BF16。NVFP4-A16 为 W4A16，不是 W4A4 |
| `foldquant-w8a8` | 上游 LLM SR / DiT SH W8A8，记录 304 个量化投影；原生 TRT INT8 plugin、完整七段模型 |
| `foldquant-awq` | 上游 ModelOpt AWQ W4A16 的 LLM/DiT 路径，其他组件保持上游浮点策略 |
| `foldquant-smoothquant` | 上游 ModelOpt SmoothQuant W8A8 的 LLM/DiT 路径，其他组件保持上游浮点策略 |

AWQ/SmoothQuant 导出需要同时重放量化检查点中的平滑权重和 quantizer 状态。早期 AWQ r2 导出因 ModelOpt/Torch 导入不兼容失败；SmoothQuant r2 在测量前中止以修正权重绑定，r3 又发现 Hugging Face 共享权重别名未被枚举。这些尝试不构成有效结果。后续使用包括共享别名的完整权重装载，以及同一 `JitScalarType` 的兼容导入别名；未改量化算法或引擎计算。该类型的新导入位置可在 [ModelOpt 上游实现](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/export_onnx.py) 核对。

### 未形成有效延迟的配方

PhyAI 图模式的 FP8-A8、block-FP8-A8、INT8-A8、INT4/FP8-A8 未通过 A/B/A 重复性检查；最大重复偏差分别为 0.0625、0.03125、0.046875、0.03125。普通 FP8-A8 关闭图后可测量，但延迟显著增加；其余三个 eager 诊断仍未通过（0.03125、0.078125、0.03125）。这里的判定是沿用协议的重复性检查，不是新增数值精度阈值，也不据此推断唯一故障根因。

FlashRT FP8 backbone + FP16 head 在当前固定版本下未响应单独改变的图像输入；该 frontend 缓存 cross-K/V 的生命周期与 fresh-backbone 路径不匹配。没有为它修改引擎缓存逻辑，也没有报告无效延迟。

NVIDIA 五条路径及 FoldQuant 三条路径最终均完成测量。前述依赖/导出失败是中间尝试，不是这些配方的最终状态；原始失败日志仍保留。SmoothQuant 最终 mean 为 26.428 ms，AWQ 为 57.523 ms，原生 FoldQuant W8A8 为 29.941 ms，不能仅按量化位数推断速度。

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

固定源码：NVIDIA `944fe70c4fd54e3fcb80a7d7258a328f9c061176`、FoldQuant `7d98edd705c33ef994b3caf8c2c93d28da78413a`、PhyAI `f983a8f84684aa7db4dfac7184b7b5c787635978`、FlashRT `1fbe995089af4148bacd10ead61764d0aeb80f7a`。FlashRT 复用已有构建版本，未因调研发现更新 main 就替换已测环境。

完整构建/运行入口及资产生命周期见 [操作手册](../../docs/guides/groot-deployment-recipes.md)。历史 FlashRT FP8/NVFP4、EmbodiInfer BF16、PhyAI BF16 与 eager reference 仍引用既有报告，本轮没有重跑。
