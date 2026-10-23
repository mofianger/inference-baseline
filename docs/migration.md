# 分阶段迁移计划

本轮整理文档、研究契约、目录和协作模板，不移动历史产物，不改变 benchmark 数学计算或计时。

| 阶段 | 工作 | 完成判据 |
|---|---|---|
| 已完成：文档与目录 | 核心职责、现有组合清单、操作手册、协作模板 | 文档可导航、示例对应实际结果、明确未实现能力 |
| 下一步：配置解析 | 给 ModelSpec/Protocol/Dataset/Recipe/Platform 增加 schema、ID 解析与冲突检查，生成现有 CLI config | 同一声明生成相同 resolved config；拒绝不兼容组合 |
| 下一步：模型插件 | 将输入构建/校验、输出 shape、噪声、参考实现、误差解释及输入响应 probe 从通用 Runner 抽出 | GR00T 继续运行；新模型不修改 Runner 主流程 |
| 下一步：Adapter 生命周期 | 明确 prepare/infer/runtime_stats/close；记录实际 Recipe 开关与初始化产物 | prepared work 不漏计；销毁后资源释放；禁止静默回退 |
| 下一步：Run 契约 | 独立 run ID、declared/resolved 配置、状态、制品 URI/hash、统计复算 | 失败也留记录；结果不可覆盖；不同配置不能冒充同一实验 |
| 下一步：协作自动化 | PR 的文档/schema/摘要/统计检查，机器任务占用约定 | 自动检查不需要访问私有权重或 GPU；GPU 实测由认领者执行 |

先用已有 GR00T + 5090 的三个引擎做兼容性回归，再引入 π0.5 或第二种硬件。代码迁移引起的输出或计时变化需要单独报告；不要用“只是重构”掩盖结果变化。

现阶段目录保留映射：`inference_baseline/adapters/` 是 Adapter，`runner.py` 是 Runner，`hardware.py` 是硬件后端，`configs/5090/` 是可执行配置；不新增与这些实现重名的空根目录。`results/` 先作为实验索引，既有数据仍在 `reports/`；实际大文件在 Git 忽略的 `datasets/`、`artifacts/`、`third_party/`。
