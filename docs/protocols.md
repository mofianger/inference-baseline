# 测试协议与可比性

## 必须明确的字段

1. 输入与输出：格式、dtype、device、形状、相机顺序、坐标/归一化语义。
2. 计算工作量：batch、视角、图像和文本长度、状态历史、去噪步数、动作块长度。
3. 时间边界：包括/排除哪些计算、传输、同步与后处理。
4. 常量与缓存：允许缓存的布局/权重相关常量；禁止复用的观测相关计算。
5. 测量流程：预热、正式次数、重复运行、设备独占、线程/时钟控制。
6. 正确性：参考实现、输出空间、指标、阈值（可以未设置）。

## 当前协议

ID：`groot-n17-b1-v3-model-only-v1`，见 [契约](../specs/protocols/groot-n17-b1-v3-model-only-v1.json)。

GPU prepared tensors 已就绪 → learned embeddings → backbone → 状态编码和 4 步去噪 → GPU normalized action `[1,40,132]` 完成。

Runner 执行 `sync → start → adapter.infer → sync → stop`。计时内等待完成，因此是同步 CPU wall latency；不是无同步的提交耗时，也不是纯 GPU kernel 时间之和。

排除外部预处理、外部 H2D/D2H、加载、校准、编译、图捕获与动作反归一化；引擎调用内部的 metadata 传输、复制、调度和同步计入。固定 token layout 下允许位置/RoPE 常量缓存；不允许复用上一观测的视觉特征。

FlashRT 适配器在计时内先计算 patch/token embedding 再构造 aux。EmbodiInfer 每次重放前缀图计算新特征，同一次请求的 4 步去噪复用 cross-K/V；完整去噪图一次重放全部 4 步及 Euler 更新。图复用不是输出缓存。

## 比较分组

同条件引擎比较至少固定模型/权重哈希、协议版本、数据集/噪声、工作量、硬件条件和参考输出。Recipe 精度与执行方式可以不同，但必须披露误差，不能只按延迟宣称优劣。

研究跨硬件差异时，可以有意改变硬件，但应把它标为实验变量；研究 batch/视角变化时亦然。不要将改变多个条件的结果描述成单因素加速比。

mean 对 mean、p50 对 p50。CPU raw E2E、GPU tensor model-only、CUDA event time 与 action-head-only 分组展示。GPU 型号相同不等于机器、软件和控制条件相同。当前 CLI 只强制相同 dataset manifest，不能代替人工完整可比性审查。

## 精度与稳态

当前比较完整归一化 132 维动作，报告 MAE、RMSE、relative RMSE、max abs、cosine 和逐维 MAE；不隐藏 padded 维度。阈值为 null，`not_configured` 不能写成“精度通过”。不同参考实现或物理动作/归一化动作的误差不直接排名。

分别检查输入变化、A/B/A、有限值、输出形状和输入未被修改。图数量/编译计数在预热后应稳定。改变权重、layout、步数或 Recipe 需重新准备，不允许把新增编译混入稳态统计。

现有 100 次调用重复 4 个合成观测，只是接入验证。正式研究应预先约定真实数据、样本规模、独立重复轮次及全部结果的汇总方式，不能跑完再选择最快轮次。
