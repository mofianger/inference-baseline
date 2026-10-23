# Run：<唯一 ID>

这是填写模板，不是已执行结果。建议 ID：UTC 时间-执行者-机器-Recipe-重复编号。

## 实验定义

- Issue / 执行者 / 日期：
- ModelSpec ID / checkpoint revision 与 hash：
- Protocol ID / 是否有偏离：
- Dataset ID / manifest SHA256 / 获取方式：
- ReferenceSpec ID / 输出空间 / 精度阈值：
- Recipe ID / 实际精度、优化开关与计算变化：
- 机器、硬件、设备独占与功耗/时钟策略：
- 计划重复次数 / 本次重复编号：

## 可复现来源

- Runner commit / 引擎 commit / tracked 与 untracked 代码变更：
- 安装脚本、环境锁、构建参数、二进制 hash：
- 校准输入、算法、种子及缓存来源：
- 完整命令 / resolved config 文件：
- 大型制品 URI、hash、访问与保留方式：

## 执行记录

- 状态：measured / failed / skipped（与精度 gate 独立）：
- 预热、正式次数、样本数及输入 layout：
- 输入/输出语义、动态输入与图状态校验：
- summary、逐调用延迟、日志及失败记录链接：
- 误差指标与 gate（未设阈值写 not_configured）：
- 原始 count/mean/p50/p95/p99 复算结果：
- 与其他 Run 比较时固定的条件、变化的变量：
- 限制、异常、重试原因与关联 Run：

禁止仅保存最好一轮、覆盖旧 Run、将失败/未设置门槛改写为精度通过。
