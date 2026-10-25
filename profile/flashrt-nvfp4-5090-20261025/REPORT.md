# FlashRT NVFP4：Kernel 与 SM–CTA Profile

[打开交互报告](flashrt_nvfp4_profile.html)。单文件内嵌全部绘图数据，可离线打开。

## 范围与结果

RTX 5090，GR00T N1.7，B=1，三视角 256×256，4 步去噪，H40。沿用 `configs/5090/flashrt_nvfp4.json` 和 `datasets/groot-n17-v3`，固定 evaluation 样本及初始噪声。FlashRT commit 为 `1fbe995089af4148bacd10ead61764d0aeb80f7a`，源码未修改。

计时从准备好的嵌入进入 FlashRT frontend 开始，到输出完成；排除 Adapter 的 patch/token embedding，保留 FlashRT 内部拷贝、转换和同步。因此本次 8.35 ms 不能直接与包含嵌入准备的既有 benchmark 比较。

| 采集 | 结果 |
|---|---|
| 未插桩，30 次同步 wall time | 中位数 8.3468 ms |
| torch.profiler，10 次请求 | GPU 时间跨度中位数 9.0740 ms；每次 1,586 个 Kernel |
| Nsight Systems，3 次请求 | GPU 跨度 10.1992 / 9.4590 / 9.1135 ms |
| NVBit，独立一次请求 | 586,187 条 CTA 记录，覆盖 170 个 SM，1,575 个计算节点 |
| NVBit 开启记录 | 66.8747 ms；注入代码但关闭记录时，中位数 59.4542 ms |

两个 profiler 对 CUDA 内部拷贝的归类不同：torch 显示 8 个内部 memcpy Kernel，Nsight Systems 的 Kernel 数为 1,578。共同部分为 1,575 个计算节点及 3 个外层转换。

## 可以从图中看到什么

- 实际执行的视觉部分为 18 层 ViT，DeepStack 取第 6、12、18 层；随后为 16 层 LLM 和 4 层 VL。动作部分包含 Cross K/V、状态编码与 4 轮去噪，每轮 32 层 DiT。
- 去噪节点存在明显重叠。例如 Step 1 / DiT 01 的代表请求跨度约 27.81 μs，单个 Kernel 时长之和会更大。阶段柱图统计 Kernel 时长之和，不能相加当作请求延迟。
- 同一 DiT 层内部分 GEMM 的 Grid 只有 36 或 72 个 CTA，少于 170 个 SM。逐 SM 图能显示该次插桩运行的分布；是否值得改变切分方式，还需要独立的 Kernel 性能实验。
- 本轮用于定位执行结构、重叠及 CTA 分布；未采集指令停顿、带宽、Tensor Core 利用率等 NCU 指标，尚不据此判定计算或访存瓶颈。

## CTA 覆盖与解释

NVBit 1.8.1 在计算 Kernel 的入口和 EXIT 前记录 GPU globaltimer 与 SM ID，以最早线程入口和最晚线程退出构成 CTA 区间。每个 CTA 独立槽位，缓冲区溢出为 0；所有已记录 CTA 的入口/退出线程数均与 Block 线程数一致。

1,573 个计算节点的记录数与 Grid 一致。Cross K/V 节点 #449、#451 的 Grid 分别为 192、384，但各记录到 170 个 CTA。捕获时的 launch 配置与设备读取的 Grid 均确认原始维度，缺口原因尚未确定。页面保留已观测记录并提示覆盖不全；这些节点的空白不可解释为空闲。

插桩改变耗时及调度。第二层用于观察插桩执行，不代表原路径的精确 SM 调度，也不能直接用它估算原路径优化收益。重复执行输出最大差为 0；这是每种采集模式内部重复性检查，没有替代跨模式或相对官方模型的精度评估。

## 复现与文件

运行环境：PyTorch 2.11.0+cu128、CUDA Toolkit 13.0、驱动 580.95.05、Nsight Systems 2025.3.2、NVBit 1.8.1。沿用 FlashRT 独立虚拟环境。

原始数据在服务器 `/data/inference-baseline/profile/flashrt-nvfp4-5090-20261025/reports/`：

- `timeline/trace.json`：第一层时间线；`timeline/metadata.json`：未插桩 wall time。
- `mapping-r2/mapping.json`：独立源码关联采集，不参与耗时计算。
- `cta-r7/cta.bin`、`cta.names.tsv`、`cta.capture.tsv`：第二层原始记录。
- `nsys.nsys-rep`、`nsys.sqlite`：Nsight Systems 交叉核对。

源码与采集入口见 [harness/run.sh](harness/run.sh)。构建网页需把上述文件分别放入本目录 `reports/trace.json`、`mapping.json`、`cta.bin`、`cta.names.tsv`、`cta.capture.tsv`，然后运行：

```bash
python profile/flashrt-nvfp4-5090-20261025/analysis/build_report.py
```

构建过程核对计算节点名称、序列、Grid、Block、线程入口/退出数及记录范围；`analysis/summary.json` 保存覆盖例外和原始文件 SHA256。大型原始文件不进 Git，HTML、汇总和采集代码进入 Git。

网页已在 Chrome 检查数据加载、层切换、Kernel 联动、单节点/整组/完整请求 CTA 视图、SM 范围筛选和手机宽度布局。
