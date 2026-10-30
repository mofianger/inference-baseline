# FlashRT NVFP4 / GR00T N1.7 / RTX 5090：执行图与 Kernel Timeline

[打开独立静态网站](flashrt_nvfp4_profile.html)

## 本次范围

沿用三视角冻结测试集 `datasets/groot-n17-v3`：B=1，3×256×256，217 个视觉/文字/特殊 token，其中图像 token 192 个，其他位置 25 个。固定初始噪声，4 步去噪，动作输出 40×132，embodiment 与既有协议一致。

FlashRT 固定为 `1fbe995089af4148bacd10ead61764d0aeb80f7a`，`use_fp4=True`、融合 FP8 主干开启，主干 NVFP4 实验路径关闭，DiT PDL 保留。未修改 upstream 源码、未开发新的引擎算子。

本次使用独立 harness 的 `CompleteAdapter` 修正 aux：由官方 Transformers `Qwen3VLVisionModel` 从同一 GR00T checkpoint 严格加载全部视觉权重，计算 24 层 ViT 和 Merger，将最终视觉 embedding 填入 LLM 输入的图像位置。该官方模块也计算 DeepStack，返回的这部分特征在当前接入方式中未使用；FlashRT 仍通过自己的 FP8 ViT18 路径计算 DeepStack。这是对已有输入契约的完整接入，含有重复视觉计算，不是优化后的融合视觉部署。

**旧版通用 `inference_baseline/adapters/flashrt.py` 尚未在本次任务中整体迁移；本报告只使用独立修正后的 harness。旧版 20261025 Profile 的 aux 缺少最终视觉 embedding，不能作为完整模型有效 Baseline。**

## 两种计时边界

| 范围 | 内容 | 30 次同步 wall 中位数 |
|---|---|---:|
| FlashRT frontend | 正确且完整的 aux 已在 GPU 就绪；包含输入转换、图内计算、内部拷贝、输出转换与同步 | 8.3244 ms |
| 完整接入调用 | 从预处理后的 GPU 模型输入开始，每次重新做官方 BF16 视觉准备、patch/token embedding、视觉位置填入，再运行 FlashRT | 21.7424 ms |

两者都不含原始图像解码、图像预处理和 H2D。第二个指标包含 Python/官方视觉实现的调度，不应描述为 FlashRT 纯引擎延迟。

另采集 10 次 torch.profiler 请求。FlashRT GPU 时间跨度中位数 8.7549 ms，完整调用 GPU 时间跨度中位数 30.0947 ms；profiler 对大量官方 eager 小算子的影响更明显，不能与未开启 profiler 的 wall time 混为一谈。页面分别显示两种结果。没有运行 NVBit，也没有使用此前扰动较大的 CTA trace。

## 执行顺序与解释

- 完整视觉 embedding → FlashRT ViT18/DeepStack → 16 层 LLM → VLLN/4 层 VL → Cross K/V → 状态编码 → 4 步动作编码、32 层 DiT、解码与 Euler 更新。
- 共 1,586 个 frontend Kernel，其中 1,575 个计算 Kernel 与独立源码映射采集的名称、顺序、Grid 和 Block 一致；其余为 8 个内部拷贝 Kernel、3 个边界转换 Kernel。
- 原始 trace 中还存在 `gpu_memcpy` 活动；页面 Kernel 视图只统计 `kernel` 类事件。同步 wall time 保留所有这些操作。
- 每个 Kernel 的详情提供逻辑输入/输出、精度、融合内容、完整名称、源文件/行号、本次时长及 10 次采集的中位/最小/最大。张量契约来自固定版本源码，不是从 GPU 指针自动反推的 shape。
- 计算图箭头表示调用顺序；PDL 允许执行区间重叠，实际数据流还包括残差和共享 Cross K/V。纵向分行仅用于显示重叠，不代表 SM 或独立 stream。

阶段卡片使用各计算组起止区间的并集，避免把 DeepStack 各 tap 之间的 ViT 计算记入 DeepStack；阶段内 Kernel 时长之和仅作辅助显示。四步去噪分别约 0.9922、0.9768、0.9798、0.9791 ms。不能再用约 4 ms 的 Kernel 时长和来描述每一步耗时。

FFN Down `(M,N,K)=(41,1536,6144)` 在 128 个调用位置各自的 10 次中位数上，再取中位数为 17.2318 μs。这是带 PDL 的图内 Kernel 区间，含等待和重叠，不是独立 GEMM microbenchmark；也不包含后续合并 partials、残差与归一化 Kernel。

## 正确性检查

当前展示样本为 `synthetic-0001`，相对冻结官方 BF16 输出：MAE 0.01027394，RMSE 0.01482826，最大绝对误差 0.390625，余弦相似度 0.99991155。

5 个样本均有误差记录，最大绝对误差的跨样本最大值为 0.880859375。按用户约定不设置通过阈值，不宣称任务精度通过。全部记录见 `analysis/summary.json`，网页包含各样本表格。

## 文件与复现

- `harness/collect.py`：官方视觉接入、同步计时、trace 和源码映射。
- `harness/site.html`：静态网站模板。
- `analysis/semantics.py`：按锁定源码维护的张量与融合说明。
- `analysis/build.py`：关联验证、统计、静态页面生成。
- `analysis/kernels.json`：逐 Kernel 记录；`analysis/summary.json`：环境、误差、汇总和 SHA256。

服务器原始数据：`/data/inference-baseline/profile/flashrt-nvfp4-5090-20261030/reports/trace/{frontend,integrated,metadata}.json`，以及 `reports/mapping/{mapping,catalog,metadata}.json`。大型原始 trace 不进入 Git。

环境使用 FlashRT 自己的 torch 2.11.0+cu128。为载入 Qwen3-VL，harness 的 `deps/` 从现有 `.venv-groot-trt/lib/python3.12/site-packages` 链接 `transformers` 4.57.3、`tokenizers` 0.22.2 及对应 dist-info；只影响该 harness，不更改 FlashRT 原环境包。部署到其他机器时需准备相同依赖。

在服务器仓库根目录，选择新的输出目录，运行：

```bash
PY=third_party/FlashRT/.venv/bin/python
RUN=profile/flashrt-nvfp4-5090-20261030
$PY "$RUN/harness/collect.py" --mode trace --out "$RUN/reports/new-trace"
$PY "$RUN/harness/collect.py" --mode mapping --out "$RUN/reports/new-mapping"
```

本地构建读取 `reports/` 下的 `frontend.json`、`integrated.json`、`metadata.json`（来自 trace），以及 `mapping.json`、`catalog.json`。下载后运行：

```bash
python profile/flashrt-nvfp4-5090-20261030/analysis/build.py
```

已用 Chrome 离线打开，验证模型跳转、逐 Kernel 详情、整体/局部时间线、缩放平移、搜索、完整调用范围切换、CSV 导出及手机宽度布局，无 JS 错误。
