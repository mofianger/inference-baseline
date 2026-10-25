# GR00T N1.7：部署配方的构建与测量

沿用 `groot-n17-v3` 的 B1、三视角 256×256、4 步、H40×A132 协议。已有成功结果不重复执行；当前只接上游已实现计算路径。结果及源码固定版本见 [本轮报告](../../reports/2026-10-25-5090-groot-new-recipes/README.md)。

## 环境与输入

服务器工作目录为 `/data/inference-baseline`。FlashRT、PhyAI 分别使用各自 `third_party/<engine>/.venv`；NVIDIA 部署使用 `.venv-groot-trt`，FoldQuant 使用 `.venv-foldquant`。所有引擎源码通过 config 的 `python_paths` 指向仓库内 `third_party`，不修改原来的参考实现环境。

当前新增环境使用 Python 3.12、PyTorch 2.9.0+cu128、Transformers 4.57.3、TensorRT 10.15.1.29、ModelOpt 0.39.0；依赖冻结记录见报告的 `environments/`。环境冻结记录应与引擎 commit 一起使用，不能仅安装引擎的顶层 requirements 后认为完全复现。服务器只有 CUDA 13 nvcc，ModelOpt 扩展和 FoldQuant 插件的编译日志一并保留。

测试集、权重和已构建 engine 不进入 Git。`configs/5090/` 中是当前服务器的绝对路径；换机器应复制配置并更改路径。原评估 manifest 不变，新的 PTQ 配方使用独立 `groot-n17-calibration-v2`（8 条合成校准样本），校准分区不含评估样本。

## 已构建配方：只执行 Runner

```bash
cd /data/inference-baseline
git pull --ff-only
bash scripts/run_5090.sh groot-trt-nvfp4 artifacts/NEW_UNIQUE_RUN_ID
# 其他配置名：groot-trt-baseline / groot-trt-optimized / groot-trt-fp8
# groot-trt-compile / foldquant-w8a8 / foldquant-awq / foldquant-smoothquant
# phyai-llm-fp8-a16 / phyai-llm-nvfp4-a16 / phyai-llm-int4-a16 等
```

配置存在不等于已测成功；先核对报告的最新状态。失败的配方不能跳过协议检查来获得延迟。

## 从校准与导出开始

构建与测量都占用同一 GPU lock；不要与其他测试并发。

```bash
flock artifacts/gpu-0.lock .venv-groot-trt/bin/python scripts/run_deployment_recipe.py \
  --config configs/5090/groot-trt-nvfp4.json \
  --assets artifacts/groot-trt-nvfp4 --output artifacts/NEW_UNIQUE_PIPELINE_ID \
  --retire-intermediates

flock artifacts/gpu-0.lock .venv-foldquant/bin/python scripts/run_deployment_recipe.py \
  --config configs/5090/foldquant-awq.json --start quantize \
  --assets artifacts/foldquant-awq --output artifacts/ANOTHER_UNIQUE_PIPELINE_ID \
  --retire-intermediates
```

`--assets` 必须与 config 的 `engine_dir` 对应。复用 asset 目录会重新生成构建文件，已有有效 engine 应先归档或改用新目录/config。`--output` 必须是不存在的新目录。`--start export/build/run` 只适用于前序资产仍存在且绑定正确的情况。

脚本保存每个阶段的命令、退出码、日志与源码/配置来源；失败就停止该配方。`--retire-intermediates` 在 FoldQuant 完成导出后先为生成的 quantized 检查点计算哈希并删除，测量完成或记录阶段失败后再回收其余 ONNX/quantized 文件，保留 engines、源权重、数据集和诊断；重做导出需要重新校准。它不会清理其他项目的数据。

## Adapter 的范围

NVIDIA/FoldQuant 导出脚本只将冻结 tensor 接到上游数据和 Policy 接口；校准算法、量化规则、ONNX 图以及 TRT runtime 使用上游函数。共享初始噪声通过上游 `init_actions` 接口注入，内部复制在计时内。optimized TensorRT 的已捕获输入缓冲也会更新，并验证图实际启用。

FoldQuant 原生量化从记录的整数权重重放；AWQ/SmoothQuant 还必须恢复量化检查点里的平滑浮点权重，再恢复 quantizer 状态。导出端为 ModelOpt 0.39 / Torch 2.9 的 `JitScalarType` 导入位置提供同类型别名，不改 symbolic 计算、精度规则或引擎源码。

PhyAI 只使用现有 Kernel Policy 配置，记录实际替换的 LLM 投影层。A8 图模式若失败，可用已有 eager 开关作为单独 Recipe 诊断；不能把两者视为同一结果。

## 检查与归档

有效 Run 必须通过输出形状/设备/finite、A/B/A 重复性、图像/state/noise 响应、输入未修改及图状态稳定检查。数值误差相对冻结 reference 报告，当前 `thresholds=null`，不能称精度通过。

`scripts/package_groot_recipe_runs.py` 是本轮专用归档器：导出逐请求延迟、summary、失败、日志摘要、资产哈希和环境快照至本轮报告。新的实验批次应使用独立 Run ID 和目录，避免覆盖历史记录；Git 只同步这些小型证据及配置，不同步权重/engine。
