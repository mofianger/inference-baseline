# 多用户与 Agent 协作

## 工作单位

一个实验任务固定“模型规格 + 协议 + 数据集/reference + Recipe + 硬件要求”，由某位用户及其 Agent 认领。任务可产生多个 Run；重复实验不覆盖旧结果。Issue 是计划和讨论，Run 是事实记录，PR 是可审查的交付。

## 推荐流程

1. 用 [Issue 模板](../../.github/ISSUE_TEMPLATE/benchmark.md) 定义任务、负责人和机器，明确计划测试哪些 Recipe，哪些不在范围内。
2. 从约定基线创建分支，例如 `bench/alice/groot-flashrt-5090`。每位用户使用独立 clone/worktree，不在共享工作目录同时修改文件。
3. 如需改 Adapter，先提交实现 PR 或明确分开提交代码与结果；固定实际测试的 commit，再执行。Agent 不得把待实现能力写成已支持。
4. 预约机器或使用真正共享的 GPU 锁。只按 Git 分支隔离代码不能隔离 GPU/CPU/显存资源。
5. 依照运行手册执行全部预定组合和重复轮次，保存失败、跳过及不兼容原因；禁止只提交最快结果。
6. 以唯一目录提交结果 PR；使用 [PR 模板](../../.github/PULL_REQUEST_TEMPLATE.md) 审查版本、输入、边界、误差、统计与限制。
7. 合并后更新 [结果索引](../../results/README.md)。结论应指向具体 Run，不能只写“某引擎最快”。

## 本地和服务器通过 Git 协作

同一个人的工作分支可以在本地编辑、commit/push，再在服务器 pull 执行，随后由服务器提交结果并 push，本地 pull。不同机器/用户不要同时写同一分支；发生分歧时正常 merge/rebase，不 force-push 覆盖别人的实验。

协议和 Recipe 的语义修改必须更新版本并写明原因。结果目录合并冲突通常意味着 Run ID 重复，保留两份并修正 ID，不手工拼接两次运行的 summary。

## 状态与资产

任务状态可用 planned / claimed / running / completed / blocked / unsupported。Run 应区分 measured、failed 和 skipped；数值门槛状态独立记录。当前 Runner 成功状态为 measured，数值 gate 可为 not_configured；不要把两者合成一个“通过”。

Git 管代码、契约、任务模板和小型结果；外部存储管权重、数据集、构建缓存及 profiler 大文件。记录 URI、hash、许可/访问限制和保留策略。只有机器本地路径的数据需说明如何让协作者获取；哈希可以验证一致性，但不能替代下载方式。

## 自动化现状

当前没有 GitHub CI、调度服务或 schema 自动检查；PR 中的核验由执行者和审查者完成。未来 CI 适合检查文档链接、JSON schema、目录/ID、摘要统计和可比性字段；不要让普通 PR 自动获取私有权重或占用远程 GPU。

可给 Agent 的任务说明：

> 按本 Issue 固定的对象和版本执行。先读 docs/README.md 和对应 Recipe；核对机器、源码和资产。任何边界或工作量变更先更新实验定义，不能静默回退。保留失败和全部重复运行，输出唯一 Run 目录；提交前核验实际配置、数值误差和逐调用统计。没有精度阈值时不声明精度通过。
