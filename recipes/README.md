# Recipe 目录

Recipe = 引擎 × 模型 × 适用硬件的一套精度/执行/校准配置。下面的 ID 是研究目录标识，当前 CLI 使用对应的 config，尚不接受 recipe ID 自动解析。

**下一轮复测入口：[5090 × GR00T N1.7 待测 Baseline Recipe 汇总](groot-n17-5090-candidates.md)**，结构化目录为 [JSON](groot-n17-5090-candidates.json)。覆盖现有三引擎的其他精度/执行路径，以及官方 TensorRT、vla.cpp、Embodied.cpp、FoldQuantVLA 候选；有延迟结果不是入表前提。待适配、需模型改造与需硬件预检均明确标记，不混入下方 measured 清单。

协作者的外部性能记录见 [V3 历史资料](../reports/2026-10-25-v3-baseline/README.md)，只作来源线索，不限制待测范围。

| 引擎 | Recipe | 状态 | 说明 |
|---|---|---|---|
| FlashRT | `flashrt-groot-n17-sm120-fp8-v1` | measured | 融合 FP8 backbone、混合 FP8/BF16 action head |
| FlashRT | `flashrt-groot-n17-sm120-nvfp4-v1` | measured | FP8 backbone、NVFP4 DiT/cross-KV |
| FlashRT | `flashrt-groot-n17-sm120-fp16-candidate` | protocol_failed | 已适配 FP8 backbone + FP16 head；图像响应检查失败，未计入有效延迟 |
| EmbodiInfer | `embodiinfer-groot-n17-bf16-graphs-v1` | measured | 前缀图、Inductor、完整 4 步去噪图 |
| PhyAI | `phyai-groot-n17-bf16-graphs-v1` | measured | 首轮已测，本轮没有重跑 |
| 官方 reference | `reference-groot-n17-eager-bf16-v1` | measured | 精确版本作为数值参考 |
| FlashRT | `flashrt-pi05-sm120-bf16-v1` | measured | BF16 完整请求图 |
| FlashRT | `flashrt-pi05-sm120-fp8-v1` | measured | FP8 prefix/decoder 混合精度 |
| FlashRT | `flashrt-pi05-sm120-nvfp4-v1` | measured | NVFP4 prefix FFN/decoder 混合精度 |
| EmbodiInfer | `embodiinfer-pi05-sm120-request-graph-v1` | measured | LeRobot 混合精度、完整请求图 |
| PhyAI | `phyai-pi05-sm120-bf16-graphs-v1` | measured | vision、prefix buckets、完整 Euler 图 |
| OpenPI reference | `reference-pi05-eager-v1` | measured | 官方 PyTorch eager 混合 BF16/FP32 |

细节：[FlashRT](flashrt/gr00t-n17.md)、[EmbodiInfer](embodiinfer/gr00t-n17.md)、[PhyAI](phyai/gr00t-n17.md)、[reference](reference/gr00t-n17.md)。

2026-10-25 新增 NVIDIA TensorRT、FoldQuant 和 PhyAI 量化路径的配置、有效测量与失败分别见 [执行报告](../reports/2026-10-25-5090-groot-new-recipes/README.md) 和 [构建手册](../docs/guides/groot-deployment-recipes.md)。候选 JSON 是调研快照，不能代替本轮实际运行状态。

π0.5 细节：[FlashRT](flashrt/pi05.md)、[EmbodiInfer](embodiinfer/pi05.md)、[PhyAI](phyai/pi05.md)、[OpenPI](reference/pi05.md)。measured 仅表示已测量，不代表数值验收通过。

当前已测硬件仅 RTX 5090。此清单不是对所有引擎版本、模型或硬件的穷尽调查；4090/Thor 的支持需要分别验证。旧 FlashRT 和无图 EmbodiInfer 保留在历史报告，不作为当前配置的同义名称。

新增条目模板：ID/版本、状态、目标模型与硬件、上游证据及核对日期、commit、精度范围、执行方式、工作量变化、校准、环境/构建、输入限制、执行 config、真实 Run 链接、未解决问题。
