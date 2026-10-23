# RTX 5090：5090-machine

服务器项目目录 `/data/inference-baseline`；单张 RTX 5090；本轮功耗上限 575 W，未锁时钟。运行时的具体驱动、GPU UUID、频率、温度和显存见每次 summary，不能用本页覆盖实际状态。

当前模型为 GR00T N1.7-3B，三视角 256px/B1/4 步/H40；CPU intra-op=8、inter-op=1、TF32 关闭。现有 [执行配置](../../configs/5090/) 含机器路径，CLI 可直接读取。

| 路径 | Python 环境 |
|---|---|
| reference | `/data/5090-machine/Isaac-GR00T/.venv` |
| EmbodiInfer | `/data/inference-baseline/.venv-embodiinfer` |
| FlashRT FP8 / NVFP4 | `/data/inference-baseline/third_party/FlashRT/.venv` |
| PhyAI | `/data/inference-baseline/third_party/phyai/.venv` |

权重只读使用 `/data/huggingface`；冻结数据集 `datasets/groot-n17-v3`。不同用户访问需要协调，仓库不分发权重。`run_5090.sh` 的锁只约束同一 checkout 内使用该脚本的测试，多用户须另行预约或配置共享锁。

[实际结果](../../results/README.md)，[运行操作](../../docs/guides/run-benchmark.md)。新增机器即使同为 5090，也要单独保存硬件和环境记录。
