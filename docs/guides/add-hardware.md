# 新增硬件平台与机器

先区分硬件类型（RTX 5090、4090、Thor）和具体机器（机器别名、CPU、GPU UUID、驱动、内存、功耗/时钟、共享状态）。同型号 GPU 不自动归为同条件实验。

## 操作步骤

1. 在 `platforms/` 描述平台能力和 Recipe 支持矩阵，不能假设另一 GPU 支持同一量化 kernel。
2. 记录 OS、CPU/架构、GPU、驱动、toolkit、依赖、编译目标、功耗与时钟策略；Thor 的 ARM 环境和统一内存不能照搬 x86 服务器。
3. 在新机器独立安装引擎、获取相同权重/数据集并核对 hash。机器路径放执行 config；凭据、token 和私有 SSH 配置不提交。
4. 核对 Adapter 所选 backend。相同 API 优先复用；不同 frontend 可使用内部选择；输入契约和输出语义改变时才拆分 Adapter。
5. 对 CUDA 通常复用 `TorchDevice`；非 CUDA 需实现正确的同步、完成判定、计时和元信息。CPU 当前主要用于协议测试，不代表已支持 CPU 上的这些 VLA 引擎。
6. 核验没有并发负载，先做正确性和稳态检查，再正式测量；提交一个机器上的真实 Run 后再标为 tested。

## 当前执行方式

`scripts/run_5090.sh` 是 5090-machine 的便利脚本，包含绝对路径，不是通用硬件调度器。其他机器使用自己的 venv 和 config 调用：

```bash
/path/to/engine-env/bin/python -m inference_baseline.cli run \
  --config configs/local-my-machine.json \
  --dataset /path/to/frozen-dataset \
  --output artifacts/UNIQUE_RUN_ID
```

`configs/local*.json` 已被 Git 忽略；成功运行的 summary 会保存 resolved config。分享前检查路径是否适合公开，不提交凭据。团队可将不含敏感信息的可复现配置另行纳入版本管理。

5090 脚本的 flock 只保护同一 checkout 内使用该脚本的进程；不同用户/checkout、直接 CLI 调用及其他 GPU 程序不受保护。多用户同机需约定共享锁位置或预约机制，不能仅看到脚本有锁就认为 GPU 已独占。
