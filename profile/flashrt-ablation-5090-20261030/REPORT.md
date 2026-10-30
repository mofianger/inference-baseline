# FlashRT 优化收益实验

结果入口：[静态结果页](ablation_results.html)、[完整结构化结果](analysis/results.json)、[Nsys 实际 Kernel 调用](analysis/kernel-dispatch.json)。

## 实验设置

RTX 5090，GR00T N1.7，batch 1，3 views × 256 × 256，4 步去噪，40 × 132 action。输入为 GPU 上就绪的完整 aux、state 和固定 noise，输出为 GPU action tensor。外部图像 embedding 和权重准备在计时外。使用同一 synthetic-0001 观测提取代表性局部输入。

FlashRT 固定为 `1fbe995089af4148bacd10ead61764d0aeb80f7a`（准确版本以运行 metadata 和 third_party checkout 为准）；实验没有修改引擎源码，拆分对照通过本目录扩展和进程内替换实现。PyTorch 2.11.0+cu128，驱动 580.95.05，Nsys 2025.3.2。

请求锁定核心 2400 MHz、显存 14001 MHz，负载期间观察到的主要实际频率为 2317 / 13801 MHz；功率上限保持 575 W。每次运行记录 GPU 遥测，结束自动恢复时钟。该频率设置与之前默认频率的测试不同，收益在本实验配对结果内计算。最终检查 GPU 已空闲。

## PDL

分别启动独立进程，关闭/开启顺序为 0、1、1、0、0、1。保持精度、Fusion、Graph 和输入一致。每次预热 5 秒，采集 150 次同步 wall latency。

| 配置 | 三轮中位数的中位数 |
|---|---:|
| PDL 关闭 | 10.853 ms |
| PDL 开启 | 9.936 ms |

节省 **0.917 ms，延迟下降 8.45%**。两组相对于参考输出的误差一致。独立 Nsys 各采集 10 次请求，每次均为 1578 个 Kernel，逐项核对 Grid 和 Block 一致。开关仍会生成不同编译变体；这里测量的是 FlashRT 的 PDL 开关效果。

## Kernel Fusion

两组关闭 PDL、开启 CUDA Graph。预分配临时缓冲，图捕获和编译在计时外。每个配置交替测 6 轮，每轮预热 2 秒、100 次同步 wall latency。

| 整条 aux → action 路径 | 中位数 |
|---|---:|
| Fused | 10.844 ms |
| Unfused（拆分 F1–F8） | 13.411 ms |

融合节省 **2.566 ms，延迟下降 19.14%**。Unfused 保留 Attention 主体等其他融合，所以此结果对应下面明确列出的拆分范围。

| 组 | 被拆分的计算 |
|---|---|
| F1 | Backbone 的残差/偏置、LayerNorm 或 RMSNorm、FP8 激活量化 |
| F2 | SwiGLU 激活与 FP8 量化 |
| F3 | DiT LayerNorm、条件调制、NVFP4 量化 |
| F4 | GEMM 分块结果归约、偏置、GELU、NVFP4 量化 |
| F5 | 分块归约、偏置、残差相加、LayerNorm、量化 |
| F6 | 分块归约、偏置、残差、下一层 AdaLN 或最终输出处理 |
| F7 | 将投影分块结果完整写出，再运行相同 Attention；融合版本直接消费分块结果 |
| F8 | 将输入量化从 skinny GEMM 内拆出，再运行相同 GEMM |

18 个代表性局部输入的 Fused/Unfused 输出逐元素一致，包括打包量化输出。整条 Unfused 路径与 Fused 的 action MAE 为 0.004441，RMSE 为 0.006567，最大误差为 0.044922；整条路径存在数值差异，尚未逐层定位全部来源。按约定只报告误差，不宣称通过精度门槛。

代表性局部热缓存估计：F2 为 3.646 → 7.117 μs，F3 为 1.913 → 4.374 μs，F4 为 2.534 → 5.080 μs，F5 为 3.290 → 6.218 μs（箭头为 Fused → Unfused）。F7/F8 只有亚微秒差距，应结合原始轮间波动解读。局部收益不累加代替整条路径实测。

## NVFP4：相同计算任务，比较高性能实现

BF16 使用 PyTorch/cuBLAS 矩阵乘，FP8 使用 `aten._scaled_mm.out`，NVFP4 使用 FlashRT skinny 或 grouped GEMM。输入输出采用相同任务边界；权重离线转换，激活在线量化、必要的分块归约、偏置与格式转换计入耗时。FP8 按接口要求补齐 M，补齐成本计入。允许不同精度数值误差。

| 任务 | BF16 fused μs | FP8 fused μs | NVFP4 fused μs |
|---|---:|---:|---:|
| Q1 Cross Q 投影 | 9.027 | 8.606 | 6.231 |
| Q2 Self QKV 投影 | 13.173 | 7.400 | 8.596 |
| Q3 O 投影 | 9.013 | 8.435 | 6.219 |
| Q4 FFN up | 16.081 | 9.016 | 9.211 |
| Q5 FFN down | 17.517 | 12.453 | 8.984 |
| Q6 文本 Cross KV | 40.052 | 24.091 | 36.283 |
| Q7 图像 Cross KV | 123.223 | 62.667 | 41.073 |
| Q8 完整 FFN | 31.262 | 22.497 | 19.950 |
| Q9 O + 残差 + LN | 13.883 | 14.295 | 8.384 |

另测 Q10 投影 + Attention，KV 长度分别为 25、192、41；结果页包含全部 12 项任务、可用的 fused/split 版本及误差。Q10 图像项使用代表性的首个 cross 输入和图像层权重，属于局部算子对照。

NVFP4 的收益取决于任务规模和融合方式。图像 Cross KV 上比 FP8 快约 34.5%，文本 Cross KV 上则比 FP8 慢约 50.6%。这体现的是各精度可用实现完成同一任务的成本，包含量化与布局处理。不同精度的“fused”表示该实现支持的融合边界；具体 Kernel 序列可查 Nsys 调用表。

## 局部计时、缓存与验证

1. 编译、分配、权重转换、CUDA Graph 捕获均在计时前完成。真实推理中提取局部输入，恢复操作防止原地更新导致输入漂移。
2. 热缓存主结果：Graph 内连续执行 64 次任务，批量首尾记录 GPU Event；减去匹配的 64 次输入恢复对照。预热 5 次 Graph 后，交替测 6 轮、每轮 10 批，报告轮中位数的中位数。保留包含恢复的耗时和恢复对照耗时，便于检查减法。这是重复执行条件下的局部耗时估计，恢复对照与任务的缓存相互作用仍可能影响估计。
3. 缓存压力对照：每个任务前写入 128 MiB 缓冲，再用任务首尾 GPU Event 计时，40 次 × 6 轮。写入和输入恢复位于计时外。该结果包含逐任务 Event 边界开销，与热缓存批量估计方法不同，不能直接相减作为缓存收益，也不代表严格冷缓存。
4. 独立运行 Nsys `cuda-hw,nvtx`，CUDA Graph 按节点采集。共核对 PDL 20、局部 Fusion 180、整条 Fusion 10、精度任务 183 个 NVTX 区间。局部 Fused 每个任务为 1 个 Kernel，拆分后包含多个 Kernel。Profiler 数据用于验证执行范围，收益使用无 Profiler 的测量。
5. 早期使用默认流的局部捕获、早期 Event 窗口和较弱 FP8 参考实现的探索结果均未用于本报告。正式数据为 `pdl/run-*.json`、`fusion-calibrated/{local,full}.json`、`precision-calibrated/precision.json`。

## 复现与数据

服务器目录：`/data/inference-baseline/profile/flashrt-ablation-5090-20261030`。原始 JSON、遥测 CSV、SQLite 和 `.nsys-rep` 保留于 `reports/`；原始数据不进入 Git，结构化摘要和代码进入 Git。

```bash
cd /data/inference-baseline
RUN=profile/flashrt-ablation-5090-20261030
PYTHON=third_party/FlashRT/.venv/bin/python
bash "$RUN/harness/pdl_run.sh"
bash "$RUN/harness/locked.sh" "$PYTHON" "$RUN/harness/fusion.py" --out "$RUN/reports/fusion-calibrated"
bash "$RUN/harness/locked.sh" "$PYTHON" "$RUN/harness/precision.py" --fixtures "$RUN/reports/fusion-calibrated/fixtures.pt" --out "$RUN/reports/precision-calibrated"
python "$RUN/analysis/summarize.py"
```

Nsys 命令见 `harness/finish.sh`。重跑时为采集文件选择新名称，保留已有数据。扩展依赖 pinned FlashRT、CUDA 工具链和 ninja，使用 `sm_120a` 编译。所有任务串行占用单卡，避免并发测试。
