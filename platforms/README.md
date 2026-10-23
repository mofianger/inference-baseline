# 硬件与机器目录

| 平台 | 当前状态 | 机器/配置 |
|---|---|---|
| NVIDIA RTX 5090 / CUDA | 已测 GR00T N1.7 | [5090-machine](rtx5090/README.md) |
| NVIDIA RTX 4090 | 本仓库尚未实测 | 上游研究数据不能替代本仓库 Run |
| NVIDIA Thor | 本仓库尚未实测 | 需核对 ARM 环境、kernel/量化支持与计时 |

新增机器记录模板：机器别名、执行者、GPU 型号/UUID、CPU/内存、OS/架构、驱动/toolkit、功耗与时钟策略、设备共享与锁、依赖安装方式、可用 Recipe、模型/数据位置及获取方式、实际 Run。不要提交凭据。

目录只是配置与能力说明；当前硬件实现仍为 [TorchDevice](../inference_baseline/hardware.py)。新设备流程见 [指南](../docs/guides/add-hardware.md)。
