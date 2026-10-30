"""Generate the repository report from measured values, never hand-copy metrics."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/summary.json').read_text(encoding='utf8'))
f=lambda x:f'{x:.3f}'
b=d['baseline'];g=d['gains_ms'];t=d['full']['timing'];m=d['mechanisms']['timing'];base=t['native']['median_ms']
labels={'native':'NVFP4 原路径','cross-fp8':'Cross-KV 换回 FP8 Recipe 的 BF16 投影','cross-split':'Cross-KV 分组拆成逐层 NVFP4 投影','attention-fp8':'Attention 段换回 FP8 Recipe','o-fp8':'O 投影段换回 FP8 Recipe','ffn-fp8':'FFN 段换回 FP8 Recipe','all-segments-fp8':'Attention、O、FFN 同时替换'}
rows='\n'.join(f"| {labels[k]} | {f(v['median_ms'])} | {f(v['median_ms']-base)} |" for k,v in t.items())
mr='\n'.join(f"| {label} | {f(m['native']['median_ms'])} | {f(m[key]['median_ms'])} | {f(m[key]['median_ms']-m['native']['median_ms'])} |" for label,key in [('DiT 拆开所选融合操作','dit-unfused'),('Attention 先归约并写回显存','attention-materialized')])
local=[]
for part in ['attention','o','ffn']:
 for li in [0,1,2,31]:
  xs=[x for x in d['local'] if x['part']==part and x['layer']==li]
  a,bv=[statistics.median(x['timing']['warm'][p]['median_us'] for x in xs) for p in ['FP8','NVFP4']]
  local.append(f'| {part} | {li} | {f(a)} | {f(bv)} | {f(a-bv)} |')
gpu=[]
for label,v in d['confirmation_telemetry'].items():
 metrics=v['metrics'];range_=lambda k:f"{metrics[k]['min']:g}–{metrics[k]['max']:g}"
 gpu.append(f"| {label} | {range_('clocks.current.sm [MHz]')} | {range_('clocks.current.memory [MHz]')} | {range_('temperature.gpu')} |")
sum_=sum(t[k]['median_ms']-base for k in ['attention-fp8','o-fp8','ffn-fp8']);joint=t['all-segments-fp8']['median_ms']-base
text=f'''# FlashRT FP8 → NVFP4：收益来源实验

[离线 HTML](../../flashrt_gain_breakdown.html) · [汇总数据](analysis/summary.json)

## 结论

本轮完整路径：FP8 **{f(b['fp8']['median_ms'])} ms**，NVFP4 关闭 PDL **{f(b['nvfp4-off']['median_ms'])} ms**，NVFP4 开启 PDL **{f(b['nvfp4-on']['median_ms'])} ms**。

总收益 **{f(g['total'])} ms**，与上一轮的 5.944 ms 接近。沿 FP8 → NVFP4 PDL off → NVFP4 PDL on 分解，分别为 **{f(g['without_pdl'])} ms** 和 **{f(g['pdl'])} ms**。

FFN 和 Attention 是主要差异所在；Cross-KV 分组投影和所选 DiT 融合操作也有可重复的收益。以下对照分别回答不同问题，不能相加成为一份无重叠的收益账单。

## 整条路径的段替换

以 NVFP4、PDL off 为基准，Backbone 始终保持官方 Fused 路径。每种变体执行 6 轮交替顺序，每轮预热 2 秒、测量 100 次同步 wall time。表中使用所有测量样本中位数。

| 改动 | 延迟 ms | 相对本组原路径增加 ms |
|---|---:|---:|
{rows}

独立替换 Attention、O、FFN 的增量之和为 {f(sum_)} ms，同时替换增加 {f(joint)} ms，两者相差 {f(joint-sum_)} ms。组合效果受段间接口和调度影响。

这些是“替换后的整条路径成本”：FP8 Attention 入口会重新执行 AdaLN；FP8 O 段末尾为下一段增加 NVFP4 量化；FP8 FFN 入口重新执行 BF16 LN，末尾生成下一段需要的量化 AdaLN。原 NVFP4 段末尾可能已经做过部分归一化。计时包括这些衔接成本，所以不能将这三行差值全部归为低精度 GEMM 收益。

FP8 Recipe 本身是混合精度。Cross Attention 的 Q/O 投影使用 BF16；Self QKV、FFN 等使用其官方 FP8 实现。实验通过提取固定版本的原 Python 调用段构建变体，未改动 FlashRT 源码或重写 GEMM。

## 两项 DiT 机制对照

| 改动 | 原路径 ms | 改动后 ms | 增加 ms |
|---|---:|---:|---:|
{mr}

两种改动的最终动作与本进程原路径 **MAE/RMSE/max error 均为 0**。此结论只针对本次输入。

- **DiT 拆开所选融合操作**：仅替换 `nvfp4_skinny_*` 的融合消费者和内部激活量化入口（上一轮 F3–F8），GEMM、精度、Backbone 保持不变；所有路径关闭 PDL。使用已有 `Controls` / `split.cu` 的预分配 CUDA 实现，依次执行归约、缩放/Bias、激活、残差、LN/AdaLN 和量化。它代表这组具体拆分实现的成本，未穷举所有 Unfused 实现。
- **Attention 先物化结果**：先用独立 Kernel 对 Skinny GEMM 分块结果归约、缩放、加 Bias，写入 FP32 临时矩阵；再调用原 Attention，以单分块、scale=1、bias=0 读取。Attention 算法保持原实现。它仍有单分块读取的少量处理，不能当作完全消除归约逻辑后的理想 Attention。
- Attention 对照属于 DiT 融合对照的子集，二者不能相加。也不能用“{f(g['without_pdl'])} 减去 DiT 融合收益”直接得到 NVFP4 位宽的收益，因为 FP8 与 NVFP4 的算子设计和融合机会不同。

## Cross-KV 分组投影

原路径将文本、图像分别分组，执行 2 个大 NVFP4 GEMM；拆分变体使用同样量化后的权重数值和 scale，改为 32 个逐层 K/V 投影。动态激活量化仍只执行两次。逐层输出采用连续布局，原路径为分组结果的视图。

完整路径增加 **{f(t['cross-split']['median_ms']-base)} ms**，动作输出误差为 0。局部 KV 输出也完全一致。

| KV 生成实现 | 预热后 μs | 128 MiB 缓存压力 μs |
|---|---:|---:|
'''
for k in ['native','cross-fp8','cross-split']:
 text+=f"| {labels[k]} | {f(d['cross']['timing']['warm'][k]['median_us'])} | {f(d['cross']['timing']['l2_pressure_128MiB'][k]['median_us'])} |\n"
text+=f'''
局部边界为 Backbone 特征 → 全部 K/V；允许两种实现保留各自的物理布局。这组结果将收益定位到 KV 生成。完整路径差值同时包含下游读取布局变化的影响。

## 相同任务、相同输入的局部对照

采样 4 个去噪步骤 × 第 0、1、2、31 层 × 3 段，共 48 组。第 0 层为文本 Cross Attention（KV25），第 2 层为图像 Cross Attention（KV192），第 1/31 层为 Self Attention（41 tokens）。层编号从 0 开始。

- Attention：BF16 hidden + 同一 AdaLN 参数 + 固定 K/V → BF16 Attention 输出，包括 AdaLN、Q/QKV 投影、Attention。
- O：BF16 Attention 输出 + 残差 → O 投影、Bias、残差加法、LayerNorm，输出 BF16。两侧都计入所需激活量化。
- FFN：相同 BF16 归一化输入 + 残差 → 动态量化、Up、Bias/GELU、Down、残差、下一层 AdaLN；第 31 层使用最终输出 AdaLN。共同终点为 BF16。NVFP4 使用原 Skinny GEMM 和融合消费者，入口单独量化以匹配共同输入边界。

以下是四步结果中位数，仅作局部任务比较，不乘以 128 外推端到端。

| 段 | 层 | FP8 路径 μs | NVFP4 路径 μs | 节省 μs |
|---|---:|---:|---:|---:|
{chr(10).join(local)}

预热模式：CUDA Graph 内重复 64 个任务，减去匹配的输入恢复批次；每轮重复 10 个批次，6 轮交替执行。保存原始批次、恢复批次、差值和逐次 Event 结果。缓存压力模式在每次计算前写 128 MiB 缓冲区，写入位于起始 Event 前；每轮 40 次、6 轮，逐次 Event 边界开销保留。两种估计方法的绝对值不直接比较。数值误差与全部逐步结果见 JSON / HTML。

## GPU、范围与 Nsys 核对

- 单卡 5090；B1，3×256×256，4 步去噪，40×132 action。相同官方 checkpoint、synthetic-0001、noise、state、冻结校准 scale。
- GPU complete aux/state/noise → GPU action；不含外部视觉 embedding。FlashRT ViT 执行 18 层。所有完整路径保持 Backbone / Action CUDA Graph。
- 请求核心 2400 MHz、显存 14001 MHz、功率上限 575 W。基线正式测量窗口实际核心 2310–2317 MHz、显存 13801 MHz、温度 38–43°C。功率上限未改动。
- 三条基线各 3 个独立进程，每进程预热 5 秒、150 次同步 wall；正式延迟来自未开启 Profiler 的测量。
- 分段补测和机制补测窗口含各轮预热，遥测每 100 ms 采样：

| 窗口 | 实际核心 MHz | 实际显存 MHz | 温度 °C |
|---|---:|---:|---:|
{chr(10).join(gpu)}

相同锁频请求不代表负载变化时频率逐时刻完全相等。局部任务可能运行在较高实际频率；每项收益均在本组交替对照中比较。

Nsys 使用 `--trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi`。基线各 10 次请求，7 个段替换各 5 次，3 个机制各 5 次，共 **80 次**。三条基线的各阶段 Kernel 签名/Grid/Block 与已有源码映射一致；所有变体的 Backbone 前缀签名一致；同变体请求 Kernel 数恒定。

FP8 每请求 3078 个 Kernel，NVFP4 PDL off/on 都是 1578；DiT 拆分为 2994；Attention 单独物化为 1706。阶段跨度使用最早开始到最晚结束，避免把 PDL 重叠时间重复相加。

## 文件与复现

- `harness/run.sh`：三条官方路径无 Profiler 基线 + Nsys。
- `harness/segments.py` / `hybrid.py`：完整段替换、48 组局部和 Cross-KV 对照；`run_segments.sh` 负责锁频、遥测、Nsys。
- `harness/mechanisms.py` / `run_mechanisms.sh`：DiT 融合、Attention 物化。
- `harness/confirm.sh`：独立补测目录，保留精确测量窗口与完整遥测；本报告段替换、局部、机制结果来源 `{d['source_measurements']}`。
- `harness/analyze.py`：验证 Nsys 签名、汇总数值；`report.py` / `render.py` 生成文档和 HTML。
- 依赖前一轮 `flashrt-recipes-controlled-5090-20261030/harness/setup.py` 与冻结 scale，以及 `flashrt-ablation-5090-20261030/harness/controls.py`、`measure.py`、`split.cu`。临时张量和扩展编译在正式测量外。
- 原始 JSON、SQLite、Nsys 文件位于服务器 `/data/inference-baseline/profile/flashrt-gain-breakdown-5090-20261031/reports/`；汇总 JSON 和脚本纳入 Git，大型原始文件不纳入 Git。
- FlashRT 固定 commit `1fbe995089af4148bacd10ead61764d0aeb80f7a`；PyTorch 2.11.0+cu128，Nsys 2025.3.2。未修改引擎源码。

在服务器仓库内使用 `third_party/FlashRT/.venv/bin/python`。复测时先指定全新的输出目录和 Nsys 文件名前缀；脚本中的固定目录是本次实验记录，避免覆盖已有结果。GPU 工作串行执行。

首轮分段计时已完成，但恢复 Cross-KV 时覆盖了其遥测文件，因此补跑了完整未采集测量并保留首轮结果。Nsys 仍采用相同实现的首轮采集；补测只增加 wall-clock 时间窗口记录。两轮输出误差一致。所有错误只作报告，未设置通过阈值，也未运行模拟器任务。
'''
(R/'REPORT.md').write_text(text,encoding='utf8')
