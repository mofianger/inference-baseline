# 新增引擎操作手册

## 1. 定义研究范围

先在 `recipes/<engine>/` 建清单：目标模型、checkpoint 兼容性、硬件、候选精度和执行路径，引用官方文档/源码并记录核对日期。Recipe 是主要推理配置，不是所有开关组合的穷举。

每项至少记录：稳定 ID/版本、源码 commit、精度覆盖范围、attention/backend、图/编译/fusion、模型计算变化、校准集和策略、支持的 shape、依赖/构建、限制、证据。状态区分 proposed、verified、measured、unsupported、blocked；“未测试”不等于“不支持”。

如 FP8 只覆盖 backbone、NVFP4 只覆盖 action head，应准确记录；不能仅凭 frontend 类名判断实际精度。跳层、token pruning、减少去噪步数要单列，不允许在同一 Recipe 下静默改变工作量。

## 2. 固定环境与源码

独立 clone 到 `third_party/<engine>`，固定 commit；使用独立 venv 或容器，提交安装脚本和版本说明。记录编译器、架构参数、动态库路径/哈希和必要校准信息。不要复制其他人的未知二进制或复用不明来源校准缓存。

源码补丁必须保留 diff 与原因；当前 revision 检查忽略未跟踪文件，因此还应人工检查真正执行的未跟踪脚本、构建文件和插件。

## 3. 实现边界适配

从 [架构](../architecture.md) 的现有 Adapter 接口入手，先明确每个计算属于初始化还是每请求推理。将模型规格约定的输入映射到引擎内部输入，返回规定的 device tensor；需要物理动作解码的协议另设入口。

通过 registry 或 `module:Class` 接入。为主要 Recipe 建独立 config，使用唯一 label；同引擎多精度结果不能共享同一个导出文件名。初始化完成后检查实际 backend、量化开关和图，不可静默降级为其他实现。

## 4. 验证后测量

- 检查 checkpoint/fixture hash、输入语义、步数、action shape、dtype 和相机顺序。
- 与固定 reference 比误差；未设置阈值就报告 `not_configured`。
- 检查新图像/state/noise 能影响输出、A/B/A 稳定、原始输入不被修改。
- 覆盖所有待测 layout 预热，确认正式阶段不新增图或编译。
- 按 [运行手册](run-benchmark.md) 留下完整 Run，公开失败和限制。

现有范例：[FlashRT](../../recipes/flashrt/gr00t-n17.md)、[EmbodiInfer](../../recipes/embodiinfer/gr00t-n17.md)。接入完成不等于所有候选 Recipe 都已测完，应保留缺口。
