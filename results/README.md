# 实验与结果索引

Run 是一次实际执行，固定模型、协议、数据集/reference、Recipe、机器与软件来源。新结果使用 `results/<唯一run-id>/`；历史 `reports/` 不移动、不覆盖。结果是测量事实，不是所有条件下的排名。

| 轮次 | 实测组合 | 记录 |
|---|---|---|
| 2026-10-23 首轮接入 | GR00T N1.7 / 三视角 / 5090：reference、旧 FlashRT、无完整图 EmbodiInfer、PhyAI | [首轮报告](../reports/2026-10-23-5090-smoke/README.md) |
| 2026-10-23 优化复测 | 同一数据集/协议：EmbodiInfer 完整图、新构建 FlashRT FP8/NVFP4 | [复测报告](../reports/2026-10-23-5090-optimized/README.md) |
| 上游对照研究 | 官方发布条件与我们的结果对照，不是新的实测 | [对照报告](../reports/2026-10-23-5090-smoke/upstream-comparison.md) |

最新一轮 mean：EmbodiInfer 28.482 ms、FlashRT FP8 14.418 ms、NVFP4 8.974 ms；数值门槛未设置，不能称为精度通过。完整误差、限制和原始记录以报告为准。

新 Run 先阅读 [运行手册](../docs/guides/run-benchmark.md)，复制 [说明模板](run-template.md)。导出文件名仍由当前 exporter 的 label 决定，不要求提前重命名历史 summary。当前没有自动索引生成或统一 Run schema 校验。
