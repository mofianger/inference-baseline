# 文档导航

协作者资料整理见 [V3 Baseline 总表](../reports/2026-10-25-v3-baseline/README.md)；当前重点为 [5090 / GR00T N1.7 / FP8 与 NVFP4](../reports/2026-10-25-v3-baseline/5090-groot-n17.md)。

π0.5 的协议、环境与复现命令见 [π0.5 / 5090 指引](guides/pi05-5090.md)，实测及数值误差见 [结果报告](../reports/2026-10-23-5090-pi05/README.md)。

仓库用于研究不同引擎、模型、Recipe 和硬件组合；目前实际支持 GR00T N1.7 与 π0.5 三视角 model-only 协议。

| 需要做什么 | 阅读 |
|---|---|
| 理解核心抽象和代码职责 | [架构](architecture.md) |
| 判断计时和结果能否比较 | [测试协议](protocols.md) |
| 调研并接入一个引擎 | [新增引擎](guides/add-engine.md) |
| 接入模型、参考实现和数据集 | [新增模型](guides/add-model.md) |
| 使用新机器或硬件后端 | [新增硬件](guides/add-hardware.md) |
| 完成一次可复现测试 | [运行手册](guides/run-benchmark.md) |
| 多用户与 Agent 协作 | [协作手册](guides/collaborate.md) |
| 查看已有组合和结果 | [实验索引](../results/README.md) |
| 安排后续代码重构 | [迁移计划](migration.md) |

`specs/`、`recipes/`、`platforms/` 是研究契约及目录索引；当前 CLI 仍读取 `configs/5090/*.json`。尚未实现自动组合配置、JSON Schema 校验或 GitHub CI。文档中的目标接口不能直接当作已有 API 调用。
