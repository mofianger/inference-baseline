# FlashRT NVFP4：Nsys aux → action 时间线

[打开静态网站](flashrt_nvfp4_nsys.html)

本次只测 FlashRT 从 GPU 上已就绪的 aux、state 和初始噪声，到 GPU action tensor 完成的路径。官方视觉 embedding、patch embedding 和文字 embedding 在计时前准备好。FlashRT 内部的精度转换、D2D 拷贝、ViT/DeepStack、LLM、VL、Cross K/V 和四步动作去噪均包含在内。它是 aux-only 测量，不与完整模型输入协议的端到端结果混用。

## 设置与结果

- RTX 5090，驱动 580.95.05；Nsight Systems 2025.3.2.474。
- FlashRT `1fbe995089af4148bacd10ead61764d0aeb80f7a`，源目录干净。
- GR00T N1.7，B=1，3 views × 256 × 256；4 步去噪，40 × 132 动作输出。
- 冻结数据集 `datasets/groot-n17-v3`，当前 evaluation 样本 `synthetic-0001`，固定初始噪声。
- CUDA Graph 与 PDL 保留。没有 NVBit 插桩，没有改动 FlashRT 算子。

| 测量 | 中位数 | 次数 |
|---|---:|---:|
| 独立进程、未开启 Profiler 的同步 wall latency | **8.3253 ms** | 50 |
| Nsys `cuda-hw,nvtx` 的 GPU 时间跨度 | **8.4916 ms** | 10 |
| Nsys `cuda,nvtx` 的 GPU 时间跨度，对照采集 | **8.8662 ms** | 10 |

wall latency 在调用前后同步，包含 CPU 提交和等待。GPU 时间跨度从首个 Kernel 开始到末个 Kernel 结束，含期间空隙。两者来自不同运行，不相减估算 CPU 开销。两次 Nsys 采集的第一条请求都有额外开销；保留所有原始值，主图选跨度最接近中位数的第 10 次 `cuda-hw` 请求（8.493934 ms）。每个节点的中位、最小和最大耗时来自 10 次对应节点。

本次显式请求 Nsys 2025.3.2 的 `cuda-hw` 模式，命令接受该选项且采集成功；报告标注实际请求的采集模式。SQLite 的通用 Kernel 表本身不能独立证明每个事件的底层时间戳来源，也不据此宣称时间戳绝对准确。

## 节点与源码对应

每个请求有 **1,578 个 Kernel**：1,575 个计算 Kernel，加 3 个输入输出精度转换。此外还有 **8 个 Graph 内 D2D Memcpy** 和 **5 个前端边界 D2D Memcpy**。

旧 PyTorch Trace 将那 8 个 Graph 内拷贝记为 `memcpy32_post` Kernel；Nsys 将其作为 Memcpy 节点记录。它们没有丢失。网站保留前版 1,586 个节点的编号，8 个 Graph Memcpy 明确标记，Grid/Block 显示为“—”。额外 5 个边界拷贝在总览下方用金色短线显示，均在计时范围内。

逻辑输入输出和融合说明复用前版源码审计。使用 GNU `c++filt` 将 Nsys 原始符号转换成与 PyTorch 一致的名称，逐项验证全部 1,578 个 Kernel 的名称、顺序、Grid 和 Block；8 个拷贝按顺序与字节数核对（每步 3,072 B 状态 token、122,880 B 动作 token）。两组采集共 20 次均通过校验。所有新页面的时长来自 Nsys SQLite，没有复用旧时长。

Nsys 诊断包含 “Not all NVTX events might have been collected” 和 “Not all CUDA events might have been collected” 提示；不能只凭工具退出成功就认定完整。此处另行验证 10 个完整 REQUEST 范围，每次均有同一完整计算序列和 13 次拷贝；原始诊断保留在 `analysis/summary.json`。这个校验覆盖被测路径，不代表采集了进程初始化阶段的全部事件。

## Gate / Up 的微小交叠

LLM 第 1 层 #211 Gate、#212 Up 在同一个 stream 顺序提交，FlashRT 的这段调用没有显式 PDL 设置。

| 采集方式 | 10 次中有交叠的次数 | 带符号的交叠范围 |
|---|---:|---:|
| Nsys `cuda-hw` | 1 | −0.256 ～ +0.096 μs |
| Nsys 常规 `cuda` | 9 | −0.096 ～ +0.800 μs |

正数为区间交叠，负数为间隔。主图第 10 次的间隔为 0.192 μs。两种方式都保留原始时间戳，没有裁剪相邻区间。结果支持采集方式影响亚微秒边界，但没有证明具体底层原因，不能把所有交叠归因于计时误差，也不能据此宣称有效矩阵计算并行。

## 数值检查

当前 evaluation 样本与冻结的官方 BF16 action 比较：MAE **0.01027394**，RMSE **0.01482826**，最大绝对误差 **0.390625**，余弦相似度 **0.99991155**。仅报告误差，没有设置精度通过阈值。本轮不重复全部五个样本的检查，不测模拟器任务。

## 复现与文件

采集入口为 `harness/collect.py`；完整命令见 `harness/run.sh`。在仓库根目录执行：

```bash
bash profile/flashrt-nvfp4-nsys-5090-20261030/harness/run.sh
```

需要服务器已有的 `third_party/FlashRT/.venv`、GR00T/Cosmos 权重、冻结数据集，以及上一轮的 `profile/flashrt-nvfp4-5090-20261030/harness/deps`（Transformers 4.57.3/tokenizers 0.22.2）和 `reports/calibration-corrected`。前端使用正确完整的 aux 校准缓存。再次采集请创建新的 run 目录并调整入口；脚本不会覆盖已有 `.nsys-rep`。

原始报告保存在本地与服务器同名目录的 `reports/`（Git 忽略）：

- `frontend-hw.nsys-rep` / `frontend-hw.sqlite`：主采集。
- `frontend-sw.nsys-rep` / `frontend-sw.sqlite`：常规追踪对照。
- `wall-metadata.json` / `nsys-metadata.json`：环境、协议、误差和 wall 样本。
- `symbols.json`：符号解码映射。

Git 保存 HTML、采集/解析脚本、`analysis/kernels.json` 和 `analysis/summary.json`。Summary 包含原始文件 SHA-256。网页验证覆盖离线打开、模型跳转、Kernel 详情、缩放平移、搜索、CSV 导出、交叠对照和手机布局。
