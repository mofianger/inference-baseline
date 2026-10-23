# Inference Baseline

具身推理的可复现测试框架。第一轮：**GR00T N1.7-3B / batch 1 / 三视角 / model-only / 单张 RTX 5090**。

协议以 checkpoint 为准：当前缓存 revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`，官方配置为 **4 个去噪步、H=40、action_dim=132**。不接模拟器，不评估任务成功率。当前按用户约定只报告数值误差，`thresholds: null`，不宣称精度验收通过。

## 计时协议

起点是 **GPU 上已经就绪的模型输入 tensor**，终点是 **GPU 上完整 normalized action tensor 已完成**。主指标为外层 Runner 的 synchronized wall time：计时前同步，调用一次完整模型，计时内等待 GPU 完成。不是 raw observation E2E，也不是仅 DiT 或单个 kernel。

- 输入包含 `pixel_values`（processor patchify 后、learned patch embedding 前）、`input_ids`、`attention_mask`、`image_grid_thw`、归一化 `state`、`embodiment_id`，以及固定初始 `noise`。
- 视觉 patch embedding、视觉编码、语言骨干、state encoder、完整去噪循环均计时。输出 `[1,40,132]`，不裁剪为机器人动作，不做反归一化。
- 不含图像预处理、tokenization、输入 staging H2D、输出 staging D2H、权重加载、校准、编译、warmup。引擎内部不可避免的同步/metadata 传输仍计入调用时间。
- 图像内容和 state 每个样本不同；prompt、token layout 固定。允许位置/RoPE 常量缓存，禁止跨 observation 复用视觉特征。
- 每个引擎从同一 checkpoint 加载权重；每次运行校验 checkpoint 和 fixture SHA256。混合精度可不同，结果记录实际实现配置。
- 初始噪声直接共享同一 tensor，而不是仅设置相同随机种子。参考实现仅对单个 action head 实例注入噪声，不全局 monkeypatch torch。
- 首轮数据是**三路独立合成图像**经过官方 Cosmos processor 的固定 tensor 集；归一化 state 为合成值。它验证模型计算和框架，不代表真实任务分布或任务精度。

## 目录

```text
inference_baseline/
  protocol.py          模型协议与输入检查
  dataset.py           固定数据集、参考输出与 hash
  runner.py            通用预热、校验、计时、统计
  hardware.py          同步/计时/硬件元信息接口
  adapters/            reference / flashrt / embodiinfer / phyai
configs/5090/          当前机器的可复现运行配置（无密钥）
scripts/              环境与运行辅助脚本
tests/                无 GPU 的核心协议/数值单元测试
reports/              可通过 Git 同步的小型验证报告
datasets/             本地大文件，Git 忽略
artifacts/            逐请求结果、缓存，Git 忽略
third_party/          引擎独立 clone 与环境，Git 忽略
```

## 5090-machine 使用

服务器仓库：`/data/inference-baseline`。模型只读取 `/data/huggingface` 的缓存，不复制到 Git。官方 reference 复用已有环境；FlashRT 使用 `third_party/FlashRT/.venv` 和独立源码构建，EmbodiInfer、PhyAI 也使用各自环境，避免依赖冲突。FlashRT 不再使用 `/data/5090-machine/FlashRT`。

```bash
cd /data/inference-baseline
git pull --ff-only

# FlashRT 独立克隆、固定版本、建立环境并编译 sm_120a kernels。
bash scripts/install_flashrt.sh

# 生成一次后固定；已存在的非空目录拒绝覆盖。
/data/5090-machine/Isaac-GR00T/.venv/bin/python -m inference_baseline.cli prepare \
  --config configs/5090/reference.json --output datasets/groot-n17-v3 --samples 5

# 逐引擎执行，避免多模型同时占 GPU。
bash scripts/run_5090.sh reference artifacts/reference-001
bash scripts/run_5090.sh flashrt artifacts/flashrt-fp8-new-run
bash scripts/run_5090.sh flashrt_nvfp4 artifacts/flashrt-nvfp4-new-run
bash scripts/run_5090.sh embodiinfer artifacts/embodiinfer-optimized-new-run
bash scripts/run_5090.sh phyai artifacts/phyai-001

# 只有相同数据集 manifest 的结果才能合并。
/data/5090-machine/Isaac-GR00T/.venv/bin/python -m inference_baseline.cli report \
  artifacts/reference-001/summary.json artifacts/flashrt-001/summary.json \
  artifacts/embodiinfer-001/summary.json artifacts/phyai-001/summary.json \
  --output reports/5090-comparison.md
```

每个运行目录必须是新的或为空。失败生成 `failure.json`，不会生成伪造的有效 summary。成功输出 `summary.json` 和逐请求 `latency.jsonl`。summary 包含 mean/p50/p95/p99、显存峰值、版本、硬件前后状态、输入与 checkpoint hash、逐样本误差和逐动作维度 MAE。

默认 10 warmup、100 个计时请求。5 个 fixture 中第一个只用于 FlashRT 校准，余下 4 个用于数值核验和循环计时；这是接入验证规模，**不是有统计代表性的 p99 基准**。正式测试应增加真实 fixture、请求数和独立重复轮次。

## 引擎接入边界

| 适配器 | 当前路径 | 注意事项 |
|---|---|---|
| reference | 官方 Gr00tN1d7 eager SDPA BF16 | 校验无缺失/意外权重；仅初始噪声注入 |
| embodiinfer | BF16 native prefix graph + Inductor denoise_step + 完整 LoopGraph | 每个请求重算前缀；4 步内复用 cross K/V；预热后禁止新增编译/捕获 |
| phyai | GR00TN17 Engine + 固定 CUDA Graph profiles | 包含 backbone 与 action head，显式传入噪声 |
| flashrt | 融合 FP8 backbone + mixed FP8/BF16 action head | action head 的 FFN/QKV 也有 FP8；包含每次 BF16 patch/token embedding |
| flashrt_nvfp4 配置 | 融合 FP8 backbone + NVFP4 DiT/cross K/V | 复用 FlashRT 适配器，结果使用独立 label；其余操作保留混合精度 |

FlashRT 的原生 aux 含 learned embeddings。适配器因此从原始 checkpoint 加载 embedding 权重，将这部分计算放回计时范围；仅常量位置表来自参考实现。其版本目前要求 H40/A132，不满足时直接报错，禁止静默改工作量。

数值核验与计时分开。每个 evaluation observation 与官方参考比较 MAE、RMSE、relative RMSE、max abs、cosine 和每动作维度 MAE。另做 A/B/A 重复输入校验、分别仅更换图像/state/noise 的输出响应检查、输入/noise 不被修改检查。完整 132 维均比较；不按任务 mask 隐藏误差。

EmbodiInfer 的“前缀图”覆盖视觉 patch embedding、ViT/DeepStack、语言编码、状态编码及观测相关 cross K/V；每次把新观测写入固定缓冲区后重放。“完整去噪图”把 4 次 denoise_step、Euler 更新和时间步推进捕获到一次图重放中，每次更新条件特征和初始噪声。CUDA Graph 复用的是执行计划及缓冲区，不是上一观测的计算结果。当前 token layout 固定，位置/RoPE 等布局常量缓存遵循同一协议。

## Git 协作

远程：`git@github.com:mofianger/inference-baseline.git`。本地编辑 → commit/push → 服务器 `git pull --ff-only` → 执行。服务器可将小型报告存入 `reports/` 后 commit/push，本地再 `git pull --ff-only`。不要同时修改同一个文件，不 force-push。大型权重、fixture、缓存与逐请求完整目录留在服务器；报告记录 manifest hash 来关联它们。

## 扩展硬件与模型

- 新 NVIDIA GPU（包括 Thor）：新增配置目录，指定该环境的 Python/引擎安装、checkpoint 和 `device`。硬件信息运行时探测，不依赖 `5090` 字符串，量化与 kernel 兼容由适配器检查。
- 非 CUDA 硬件：扩展 `hardware.TorchDevice` 的同步、计时与元信息实现。不能用无同步的异步提交时间替代模型延迟。
- 新引擎：实现 `(config, protocol, samples)` 构造、`infer(sample) -> device action tensor` 和 `metadata`；可在 registry 注册，也可使用 `module:Class` 插件路径。
- 新模型/π0.5：新增模型专属协议解析和 fixture builder，复用 Runner 与 metrics。当前不假装已实现 π0.5。
- 新协议（CPU tensor E2E、raw observation E2E 等）：使用新的协议名称、边界、数据集 manifest 和计时策略，不能与当前协议混合排名。

## 本地检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q inference_baseline
```

初始引擎调研、原始数据和离线网站保留在仓库根目录。它们是公开资料汇编，不要与本仓库新生成的实机结果混淆。

## 已完成的 5090 接入验证（2026-10-23）

四条路径均已实际执行；7 项核心/fixture 测试通过。服务器提交报告后，本地已通过 `git pull --ff-only` 拉回，验证了双向 Git 协作。详见 [本轮报告](reports/2026-10-23-5090-smoke/README.md)。

| 路径 | model-only p50 / ms | 平均 MAE | 最大绝对误差 |
|---|---:|---:|---:|
| 官方 eager BF16 | 82.964 | 0 | 0 |
| FlashRT mixed FP8/BF16 | 16.540 | 0.027105 | 1.296875 |
| EmbodiInfer native BF16，无完整去噪图 | 66.626 | 0.001795 | 0.126953 |
| PhyAI BF16 + native FP32 norms，CUDA Graph | 26.753 | 0.003726 | 0.566406 |

平均 MAE 按 4 个不同 evaluation 样本计算，排除额外的 A/B/A 重复检查；最大误差遍历全部 action 元素。**没有设置精度门槛，不应将这些运行称为精度通过，也不能仅按延迟选型。**这是一轮合成输入接入验证，不是各引擎最优配置排行榜。
