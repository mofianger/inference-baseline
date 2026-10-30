# FlashRT 官方 Recipe：受控基线与执行图

[打开单文件 HTML](flashrt_recipe_execution.html) · [结果摘要](analysis/summary.json) · [对照校验](analysis/validation.json)

## 结果

| 官方 Recipe | Backbone | 无 Profiler wall 中位数 | 三轮中位数 ms | Nsys 每请求 Kernel 数 |
|---|---|---:|---|---:|
| NVFP4 | Fused | **9.942 ms** | 9.933 / 9.942 / 9.944 | 1578 |
| NVFP4 | Unfused | **11.533 ms** | 11.533 / 11.532 / 11.533 | 2089 |
| FP8 混合精度 | Fused | **15.886 ms** | 15.893 / 15.886 / 15.885 | 3078 |
| FP8 混合精度 | Unfused | **17.485 ms** | 17.470 / 17.486 / 17.485 | 3589 |

官方 Fused Backbone 在 FP8 Recipe 中节省 **1.599 ms（9.15%）**，在 NVFP4 Recipe 中节省 **1.590 ms（13.79%）**。两个完整 Recipe 的差距同时包含 DiT 精度、GEMM/Attention 实现、Cross KV 和 PDL 等变化；独立精度效果继续参考前一轮 Q1–Q10 局部实验。

## 测量条件

- 单 RTX 5090、GR00T N1.7-3B，batch 1，3 views × 256×256，4 步去噪，40×132 action。
- 相同官方 checkpoint、冻结 synthetic-0001 evaluation 输入、state 和初始噪声。完整 aux 提前准备，边界为 GPU complete aux/state/noise → GPU action；外部图像 embedding 不计时。
- FlashRT 固定 `1fbe995089af4148bacd10ead61764d0aeb80f7a`。原仓库源码不变；使用现有官方路径，不加载 F1–F8 自定义拆分算子。
- 全部开启 Backbone 与 Action Head CUDA Graph。NVFP4 保持 PDL 开启，默认 FP8 使用其原有 DiT 实现。
- 请求核心 2400 MHz、显存 14001 MHz；功率上限保持 575 W。采样间隔 100 ms，wall 测量期间实际核心 **2302–2317 MHz**、显存 **13801 MHz**、温度 **38–43°C**，P1。按 Recipe 的完整读数见 summary.json。
- 锁频请求相同，但实际核心频率仍有约 0.65% 的范围差异；结果应在该实测条件下解读，不能声称逐时刻频率完全相等。结束后恢复默认频率，原始遥测留存。
- 每组 3 个独立进程，预热 5 秒，每轮 150 次前后同步 wall latency。顺序为 A/B/C/D、D/C/B/A、A/B/C/D；报告三轮中位数的中位数。CUDA 编译、权重准备、校准、Graph 捕获在计时前完成。请求之间不人为清空缓存。

## 校准如何固定

Backbone 共用此前正确完整 aux 生成的校准缓存，固定样本和随机种子。默认 FP8 DiT 会依据初始化时的 Backbone 输出重新校准，所以仅设置同一随机种子不足以保证 B/C scale 一样。

`harness/scales.py` 在官方初始化结束、Graph 捕获之前，将 Fused 组生成的 FP8 DiT 标量保存并用于两组；NVFP4 对照同样复用 Cross KV 的 `a_gs` 与关联 `alpha`。只改变校准标量的来源，不替换推理算子。冻结文件位于服务器 `reports/shared-scales/`，配对 hash 已验证一致。

## 官方开关具体改变了什么

`FLASHRT_GROOT_N17_BB_FUSED=0/1` 选择 FP8 Backbone 的 `pipeline_rtx_fp8.py` / `pipeline_rtx_fp8_fused.py`。NVFP4 Frontend 继承同一 Backbone 选择机制。

| 部分 | Unfused | Fused |
|---|---|---|
| 残差 / Bias / Norm / 量化 | 多个调用，写出中间结果 | 在生成下一次 GEMM 输入的 Kernel 中融合 |
| ViT / LLM QKV | 三个投影 | 一个合并 QKV GEMM |
| QKV 后处理 | 分步执行适用的 Bias、Norm、RoPE | 合并到 QKV 拆分 Kernel |
| GELU / SwiGLU | 与后续量化分开 | 与后续量化融合 |
| LLM Attention | FP16，K/V 展开到 16 个头 | BF16 FlashAttention-2 GQA，K/V 保留 8 个头 |
| DeepStack 调度 | ViT 后的独立阶段及中间处理 | 与 ViT tap 衔接的融合阶段及散射 |

这是官方优化路径整体收益，包含 Attention 实现和数据精度的变化。它补充了此前自定义 F1–F8 拆分实验，两者的拆分范围和对照实现不同。

## 执行图与 Timeline

页面包括四组完整 Timeline、FP8 与 NVFP4 各自的 Backbone 对照，以及 FP8 Fused/Unfused 按阶段和层展开的执行图。每张卡显示源码 API、计算内容、输入输出说明、实际维度参数，以及对应原生 Kernel 和耗时；可定位到时间线。

时间线采集使用 Nsys 2025.3.2，`--trace=cuda-hw,nvtx --cuda-graph-trace=node`。每组采集 10 次完整请求，选择 GPU 首末 Kernel 跨度最接近组中位数的一次，保留真实时间戳和重叠，不拼接各 Kernel 中位数。

源码映射另开进程：在 eager API 调用外围加 NVTX 并同步，记录调用栈、代码和维度。映射时间不用于性能报告；通过原生名称、Grid/Block 和执行序列匹配到正式 Graph 采集。已映射所有内部计算 Kernel，各组剩余 3 个为前端输入/输出类型转换，标为 Bridge。第一版映射遗漏独立 MLP runner，最终使用 `mapping-v2`，未重复性能实验。

| 阶段 | FP8 Fused Kernel 数 | FP8 Unfused Kernel 数 |
|---|---:|---:|
| ViT | 183 | 432 |
| DeepStack（含散射/初始化） | 18 | 30 |
| LLM | 192 | 403 |
| VL | 50 | 89 |
| Cross KV | 99 | 99 |
| State | 5 | 5 |
| 四步动作计算 | 2528 | 2528 |
| 前端转换 | 3 | 3 |

同一 Recipe 的 Fused/Unfused 四步动作计算 Kernel 名称、Grid、Block 序列完全一致：FP8 为 2528 个，NVFP4 为 1120 个。Backbone 关闭融合增加 511 个 Kernel。

DeepStack 在 Fused 中与 ViT 交错，其首末跨度包含其他阶段计算，不能将各阶段首末跨度相加。Kernel 耗时之和也可能因 PDL 重叠而超过实际 GPU 跨度。性能收益采用独立 wall 实验。

## 数值误差

| 配对：Unfused 相对于 Fused | Action MAE | RMSE | 最大绝对误差 |
|---|---:|---:|---:|
| FP8 | 0.002898 | 0.004859 | 0.031250 |
| NVFP4 | 0.004618 | 0.006929 | 0.138672 |

输出均为有限值；未设置验收门槛。各 Recipe 相对于冻结参考 action 的误差也在页面和 summary.json 中记录。本次为单个合成观测的接入和性能对照，不接模拟器，不代表任务成功率。

## 复现

服务器工作目录为 `/data/inference-baseline`。原始 JSON、输出 tensor、遥测 CSV、SQLite、`.nsys-rep` 均保留在本 run 的 `reports/`，不进入 Git。代码、静态 HTML、去重后的执行数据与校验摘要进入 Git。

```bash
cd /data/inference-baseline
R=profile/flashrt-recipes-controlled-5090-20261030
bash "$R/harness/run.sh"
bash "$R/harness/map.sh"
python "$R/analysis/make_template.py"
python "$R/analysis/build.py"
third_party/FlashRT/.venv/bin/python "$R/analysis/validate.py"
```

重跑请复制为新的 run 目录或更换输出文件名，避免覆盖原始采集。已有 `.nsys-rep` 名称会使 Nsys 拒绝覆盖。性能实验与源码映射串行运行，不与其他 GPU 任务并发。

验证包括：仓库 10 项单元测试通过；四组各 3 轮 wall、10 次 Nsys 请求完整；配对 scale hash 一致；DiT Kernel 序列一致；全部时间戳有效；页面缩放、阶段/层选择、源码详情和窄屏布局检查。
