# Inference Baseline

测试不同推理引擎在不同模型、Recipe 和硬件上的延迟与数值误差，并支持通过 Git/GitHub 协作执行和审查实验。

核心抽象：**模型规格、测试协议、冻结测试集、引擎 Recipe、硬件后端和运行记录（Run）**。Adapter 实现约定边界之间的模型计算，通用 Runner 负责预热、校验、同步计时和记录；ReferenceSpec 固定数值比较的依据。

当前已实际支持 **GR00T N1.7-3B / B=1 / 三视角 256×256 / 4 步去噪 / H40 / 单张 RTX 5090**。计时为 GPU prepared tensor 到 GPU normalized action 完成的同步 wall time，包括 learned embeddings，不含外部预处理、传输和动作解码。当前为合成输入接入验证，不接模拟器、未设数值验收阈值。

同时支持 **官方 π0.5 base / B=1 / 三视角 224×224 / 10 步 Euler / H50×A32 / RTX 5090**；已接入 OpenPI、EmbodiInfer、PhyAI、FlashRT BF16/FP8/NVFP4。见 [π0.5 操作指引](docs/guides/pi05-5090.md) 和 [延迟及误差报告](reports/2026-10-23-5090-pi05/README.md)。

## 从哪里开始

下一轮复测以 [RTX 5090 × GR00T N1.7 待测 Recipe 清单](recipes/groot-n17-5090-candidates.md) 为入口：列出各引擎精度/执行配方、源码证据与接入缺口，不要求已有延迟结果。协作者 V3 的 [历史性能资料归档](reports/2026-10-25-v3-baseline/README.md) 保留供追溯。

该清单中已实现路径的执行记录见 [新增 Recipe 测试报告](reports/2026-10-25-5090-groot-new-recipes/README.md)：保留有效延迟、失败诊断和协议不兼容项，已测路径没有重跑。新框架的构建入口见 [部署配方操作手册](docs/guides/groot-deployment-recipes.md)。

| 目标 | 入口 |
|---|---|
| 理解职责与当前实现边界 | [架构](docs/architecture.md)、[协议与可比性](docs/protocols.md) |
| 新增推理引擎与 Recipe | [引擎接入指南](docs/guides/add-engine.md)、[Recipe 清单](recipes/README.md) |
| 新增模型和测试集 | [模型接入指南](docs/guides/add-model.md)、[研究契约](specs/README.md) |
| 新增机器/硬件 | [硬件指南](docs/guides/add-hardware.md)、[平台目录](platforms/README.md) |
| 执行一次测试 | [运行操作手册](docs/guides/run-benchmark.md) |
| 多用户和 Agent 协作 | [协作手册](docs/guides/collaborate.md)、[实验任务模板](.github/ISSUE_TEMPLATE/benchmark.md) |
| 查看结果 | [实验索引](results/README.md)、[优化复测](reports/2026-10-23-5090-optimized/README.md) |

## 目录与职责

```text
docs/                    架构、计时规则、操作手册、迁移计划
specs/
  models/                权重与模型语义
  protocols/             测量边界与工作量
  datasets/              冻结输入、来源与哈希
  references/            数值参考实现与输出空间
recipes/                 引擎 × 模型 × 硬件的推理配置清单
platforms/               硬件能力、机器环境和控制条件
configs/5090/            当前 CLI 可直接执行的机器配置
inference_baseline/
  adapters/              各引擎的模型适配
  runner.py              预热、校验、计时和记录
  hardware.py            同步、计时与硬件元信息
  *protocol.py, *dataset.py GR00T / π0.5 协议与输入实现
scripts/                 安装、运行和报告导出
results/                 实验索引、Run 模板及后续唯一 Run 目录
reports/                 保留既有实测与上游对照记录
tests/                   核心/fixture 测试
.github/                 Issue 和 PR 协作模板
third_party/             独立引擎源码/环境，Git 忽略
datasets/, artifacts/    大型输入与原始运行目录，Git 忽略
```

`specs/` 中的 JSON 当前是描述性契约，**尚未接入自动配置解析**；CLI 仍读取 `configs/` 和 frozen dataset manifest。模型分派支持 GR00T 与 π0.5，其他模型仍需增加协议、数据集及 Adapter，不宣称已经支持任意模型或硬件。代码解耦的后续步骤见 [迁移计划](docs/migration.md)。

## 在现有服务器运行

```bash
cd /data/inference-baseline
git pull --ff-only
# 将 UNIQUE_RUN_ID 替换为本次唯一 ID；不要复用历史输出目录。
bash scripts/run_5090.sh embodiinfer artifacts/UNIQUE_RUN_ID
# 可选：flashrt / flashrt_nvfp4 / phyai / reference；串行执行。
```

各引擎独立环境，FlashRT 来自 `third_party/FlashRT/.venv`，不使用旧 FlashRT 仓库。完整安装、预检、导出和失败处理见 [运行手册](docs/guides/run-benchmark.md)。默认 10 次预热、100 次计时只用于接入验证，不代表有统计意义的任务性能或 p99。

## 已有结果

面向阅读与分享的单文件页面：[5090 × GR00T N1.7 实测结果与 Recipe 解释](groot_n17_5090_results.html)。下载后直接用浏览器打开，无需联网；包括测试设置、模型结构、17 条实测结果，以及各 Recipe 的精度分配和加速方法。更新数据后运行 `python scripts/build_groot_results_html.py` 重新生成。

GR00T 同协议优化复测 mean：EmbodiInfer **28.482 ms**、FlashRT FP8 **14.418 ms**、FlashRT NVFP4 **8.974 ms**。最大绝对误差分别为 **0.1602、1.3008、1.3438**；所有精度 gate 均为 `not_configured`。详情、原始记录与历史比较见 [报告](reports/2026-10-23-5090-optimized/README.md)。π0.5 独立记录在 [π0.5 报告](reports/2026-10-23-5090-pi05/README.md)。

仓库远程为 `git@github.com:mofianger/inference-baseline.git`。多人使用独立分支/工作目录，以唯一 Run 目录和 PR 汇总；权重与大型输入不进入 Git，实际实验记录必须携带资产哈希和获取方式。

## 验证与早期调研

```bash
python -m unittest discover -s tests -v
python -m compileall -q inference_baseline
```

早期公开资料保留原位置以兼容既有链接：[引擎清单](01_engine_inventory.md)、[平台与模型调研](02_pi05_gr00t_n17_benchmarks.md)、[研究数据](research_data.json)、[离线网站](inference_survey.html)。它们是公开资料汇编，不等于本仓库实测。
