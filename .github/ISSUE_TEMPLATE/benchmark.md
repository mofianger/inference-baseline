---
name: Benchmark experiment
about: 定义并认领一个可复现的推理测试任务
title: "[Benchmark] 模型 / 引擎 Recipe / 硬件"
---

## 研究问题与范围

希望回答什么？哪些变量固定，哪些变化？列出要测的 Recipe，以及明确不在范围内的内容。

## 固定契约

- ModelSpec / checkpoint：
- Protocol / 计时边界：
- Dataset / manifest hash / 获取方式：
- ReferenceSpec / 数值指标与阈值（可不设置）：
- 引擎 commit / 候选 Recipe / 上游证据：
- 硬件与机器要求 / 软件环境：
- 预热、正式次数、独立重复次数与统计方式：

## 认领与执行

- 负责人 / Agent / 分支：
- 机器与预约或共享锁：
- 当前状态：planned / claimed / running / completed / blocked / unsupported
- 预期产物：实现与环境配置、唯一 Run 目录、summary、逐调用记录、误差、失败说明。

## 完成条件

- [ ] Recipe 范围和实际开关明确，无静默降级
- [ ] 输入/权重/协议一致，计时边界经过核对
- [ ] 数值误差和结构检查独立报告
- [ ] 所有计划组合/重复轮次有结果或明确失败/跳过原因
- [ ] 结果可追溯并关联 PR，限制已说明
