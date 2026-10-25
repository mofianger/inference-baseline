"""Render the pending-recipe catalog; no GPU or third-party packages needed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def main():
    data = json.loads((ROOT / 'recipes/groot-n17-5090-candidates.json').read_text(encoding='utf-8'))
    rows, sources = data['recipes'], data['sources']
    assert len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        assert r['latency'] is None and r['run_status'] == 'pending'
        assert all(k in sources for k in r['evidence'])
        if r['existing_config']:
            assert (ROOT / r['existing_config']).is_file()
    counts = {g: sum(r['group'] == g for r in rows) for g in ('main', 'ablation', 'preflight')}
    lines = ['# RTX 5090 × GR00T N1.7：待测 Baseline Recipe 清单', '',
      '核查日期：2026-10-25。本页用于规划下一轮复测，**不以已有延迟决定是否入表，也不是已支持/已跑通清单**。没有在本次调研运行 GPU benchmark。', '',
      f"覆盖 7 个实现家族：FlashRT、EmbodiInfer、PhyAI、NVIDIA 官方 PyTorch/TensorRT、vla.cpp、Embodied.cpp、FoldQuantVLA/TensorRT。共 {len(rows)} 条待办：{counts['main']} 条主比较候选、{counts['ablation']} 条执行消融、{counts['preflight']} 条需先改造或硬件预检的实验项。已有实现不等于已在 5090/三视角/base checkpoint 验证。", '',
      '结构化维护源：[groot-n17-5090-candidates.json](groot-n17-5090-candidates.json)。每条包含稳定 ID、描述、入口、状态、优先级、源码证据和剩余工作；新一轮 run_status 全部为 pending，latency 全部为空。历史 Run 只作为接入证据，不填入这张待测表。', '',
      '## 固定测试约定', '',
      '统一 RTX 5090、官方 GR00T N1.7-3B base revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`、B1、三视角 256×256、4 步去噪、完整 H40×A132 normalized action；遵循 [现有协议](../specs/protocols/groot-n17-b1-v3-model-only-v1.json)。计时从 prepared GPU tensor、learned embeddings 之前，到 GPU action 完成；完整骨干、全部去噪、内部拷贝/同步均计时。', '',
      '量化资产允许由同一源 checkpoint 派生；必须记录原权重 hash、量化脚本/配置与输出资产 hash。GGUF/TensorRT/量化 checkpoint 的二进制哈希不同是正常的，但不能换成 LIBERO/SO101 微调权重。复用同一评估 fixture 和初始 noise；需额外校准数据时建立共享 calibration-only 集，不与 evaluation 混用，记录其 hash 和各配方算法。', '',
      '当前不接模拟器、只报告数值误差、无通过阈值。不得减少去噪步数/动作长度、跨观测缓存特征或更换蒸馏模型。不同 source 版本里的默认 LLM/DiT 层数不能覆盖 checkpoint config；尤其新官方导出器与旧 base 权重的兼容需预检。', '',
      '## 状态与复测顺序', '',
      '- **已接入·复测**：我们有 Adapter/config；本轮仍要新建 Run。',
      '- **上游实现·待适配**：GR00T 路径/入口存在；需接统一输入边界、noise 和输出，验证 5090。',
      '- **policy 可表达 / 通用算子·需接线**：是本次明确提出的候选配方；不是上游已交付的 GR00T preset。',
      '- **硬件预检 / 实验**：先验证编译、真实选核和边界；不承诺可运行或更快。',
      '- **P0**：先复跑已有四条引擎配置和官方 reference，再接官方 TRT 的四个 full-pipeline 配方。**P1**：PhyAI 骨干量化、EmbodiInfer 量化接线、其他 runtime。**P2**：补充精度、执行消融和高风险移植。', '',
      '## 主比较候选：引擎 × Recipe', '',
      '同一精度的不同 runtime/算法也可构成不同 Recipe。这里刻意同时保留 BF16、FP8、NVFP4、INT8、INT4 和 GGUF weight-only 对照，服务于之后编写 NVFP4 高效路径，而不是只挑低比特名称。', '']
    for group,title in [('main',None),('ablation','## 执行方式消融'),('preflight','## 先改造或硬件预检，再决定是否实测')]:
        if title:
            lines += ['',title,'']
        lines += ['| ID / 优先级 | 推理引擎 × Recipe | 配方描述 | 上游入口 / 开关 | 接入状态与缺口 |',
                  '|---|---|---|---|---|']
        for r in rows:
            if r['group'] != group:
                continue
            refs = ' '.join(f"[{key}]({sources[key]['url']})" for key in r['evidence'])
            vals = [r['id']+' / '+r['priority'],r['name'],r['description'],
                    '`'+r['entry']+'`',r['status']+'：'+r['remaining_work']+' '+refs]
            lines.append('| '+' | '.join(map(cell,vals))+' |')
    lines += ['', '## 对三个已接入引擎的核查结论', '',
      '### EmbodiInfer：已有量化组件，GR00T 尚未接成量化 preset', '',
      '固定版本的 `_build_gr00t` 只显式接收 attention、native_inference、prefix_cuda_graph 等参数，额外参数流向 VLAPolicyConfig；不能直接传 `quantization="nvfp4"` 就认为生效。`models/linear.py` 的 quantize_linear / FP8Linear / INT8Linear / NVFP4Linear 才是可复用接点。E02–E05 为本次提出的模型接线方案，首先只选 LLM/DiT 普通矩阵投影，保留 category-specific、vision、norm、输出等敏感/特殊模块，并保存具体层白名单。', '',
      'FP8 native 走原生 scaled GEMM，Triton FP8 为 weight-only；NVFP4 native 要求相应 PyTorch scaled-mm API 和布局支持。`backend=auto` 可回退 weight-only 浮点执行，故性能比较应显式选择 backend，打印实际分派，不能用“有 NVFP4 权重”证明执行了 NVFP4 tensor-core kernel。量化候选先不沿用 Inductor 开关；校验新算子后再尝试图捕获，编译支持作为另一个实验变量。', '',
      '### PhyAI：骨干可走 policy；action head 有明确接入缺口', '',
      'GR00TN17Qwen3VLBackbone 向共享骨干传 `prefix="backbone.model"`，下层继续构造非空 prefix。P02–P08 的初始范围明确限定为 LLM 投影，保持 vision/action head 原精度，便于先验证一组可控的量化对照；后续可另列包含 vision 的 scope 扩展。PI0.5 示例 YAML 的层名不能直接复制，必须枚举实际 Linear.prefix 并审计命中层和 tensor 格式。', '',
      '**关键限制：GR00TN17Linear 和 category-specific 子 Linear 以空 prefix 构建，而 LinearBase 只有 prefix 非空才调用 quant plan.resolve。** 因此仅增加一条 action_head YAML 规则不会让 DiT 自动量化。P10 必须先补稳定 prefix/加载映射，再验证模型数值、形状和 kernel 支持。', '',
      '**NVFP4 A16 与 NVFP4 A4 要区分。** 已核查的 Humming preset 是 NVFP4 block16 权重、input=null，即 W4A16。不能把 P05 标成 FlashRT 式 W4A4；通用 Nvfp4Spec 的存在也不能替代 GR00T 原生 W4A4 的装载/校准/执行验证。候选 YAML 应设置 fallback=error，并记录实际 kernel ID。', '',
      '### FlashRT：除了 FP8/NVFP4，还有有意义的混合精度和执行消融', '',
      'F03 为真实存在的 FP8 backbone + FP16 action-head frontend；F04 可拆出 cross-K/V 量化贡献；F06 可检查 PDL；F05/F07 分别检查 backbone 融合与 action graph。F08 是上游明确标为实验性的 NVFP4 backbone。直接 FP16 frontend 的 backbone 默认在 set_prompt 计算（F09），若只计 infer 会漏掉完整观测编码，必须修正边界后才能纳入。类名 FP16 也不能隐藏 FP8 装载后反量化的权重来源。', '',
      '## 新增实现的重点区别', '',
      '- **官方 TensorRT** 是 Isaac-GR00T 的完整 VLA 集成，不是 TensorRT-LLM。N06 与 FlashRT 的混合精度分配不同：官方还量化 LLM 为 NVFP4，但保留 o/down FP8；DiT 只将 FFN 设为 NVFP4。源码提供通用构建入口，但已公开 mixed-NVFP4 验证偏 Thor，本次不宣称 SM120 已通过。',
      '- **vla.cpp 与 Embodied.cpp** 都有完整 GR00T 路径。GGUF Q8/Q4 是存储与 GEMM 配方，不等同 FP8/NVFP4。Embodied.cpp 的 action loader 对量化矩阵保留原类型，不对齐时可回退 BF16；需导出 resident dtype 清单，而非只看 --outtype。两者的现有公开接口往往带 CPU 图像/后处理，需桥接 prepared GPU tensor 和完整 normalized GPU 输出。',
      '- **FoldQuantVLA** 有 GR00T N1.7 的 TensorRT INT8/INT4 及 ModelOpt 对照方案。W4A4 的 s4 kernel 支持表只列 SM75/80/86/87/89，SM120 不在表内；Q04/Q05 先预检，不按 Ada 数据预设可高速执行。Q02 的 W4A16 使用不同 groupwise plugin，不应与 s4 W4A4 混为同一硬件结论。', '',
      '## 额外配方维度与当前不纳入项', '',
      '首轮上述独立行优先。第二轮可展开 PhyAI E5M2、MXFP4 A16、INT4+INT8 A8；Embodied.cpp Q2_K/Q3_K/Q5_0/Q5_K；FoldQuant 不同旋转/GPTQ 舍入与单塔量化；官方 TRT 不同 export-mode × precision。它们的通用选项有源码依据，但需先通过对应主路径，避免把任意参数笛卡尔积当作已验证 Recipe。', '',
      'Realtime-VLA/APXInf 的本次检索未建立完整 GR00T N1.7 路径证据；llama.cpp/vLLM/SGLang/TensorRT-LLM 单独的 LLM 能力不构成完整 VLA recipe。已有 VLA 集成的 C++ 和官方 TRT 已单列。Intel [OpenVINO GR00T](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_gr00t_n1d7.html) 不在 RTX 5090 CUDA Setting 内。一步蒸馏、跨观测缓存与修改权重架构的方案不满足当前协议。这里的未纳入表示证据或协议不足，不表示项目永久不支持。', '',
      '## 开始实测前，每条 Recipe 要补齐什么', '',
      '1. 将候选落实为独立 config/Adapter，记录分模块 W/A/KV、真实 backend 和 fallback；上游命令/伪代码不是本仓库可直接执行的 CLI。',
      '2. 核对同一源权重、三视角 layout、4 步、40×132、公共 noise；转换/校准不得更换微调 checkpoint。',
      '3. CPU 原始输入接口需拆出或桥接 GPU 模型边界；TensorRT 分段组件及 PyTorch glue 合并计时；嵌入和每次观测的特征不移到计时外。',
      '4. 数值、A/B/A、输入敏感性和输入不变性检查后，确认 warmup 后无新编译/capture，再开始 100 次初测；无数值阈值仍报告误差。',
      '5. 每个候选新建 Run 并回填配置、来源提交、资产 hash、实际精度/图/kernel 和结果；不把失败预检填成延迟 0。', '',
      '## 源码快照与证据', '',
      '检索包括 GR00T N1.7 inference/quantization/TensorRT、EmbodiInfer/PhyAI 量化，以及其他完整 VLA runtime。对下列 7 个仓库建立隔离只读副本并核对代码；未修改服务器引擎环境。源码副本位于 Git 忽略的 artifacts/recipe-audit，文档用固定提交链接；JSON 的 sources 另保存每个核查文件的 SHA256。', '',
      '| 实现家族 | 本次核查提交 |', '|---|---|']
    for key,v in data['versions'].items():
        lines.append(f"| {key} | [{v['commit']}]({v['repository']}/tree/{v['commit']}) |")
    lines += ['', '历史已测 FlashRT/EmbodiInfer 提交与本次调研 main 快照不同；旧 Run 保留旧来源，不因本次发现新代码而修改。PhyAI 本次核查提交与此前固定版本相同，action-head prefix 问题因此也适用于当前接入版本。', '',
      '更新 JSON 后运行 `python scripts/render_groot_recipe_catalog.py` 重建本页。新增/变更 Recipe 保留稳定 ID；复测结果继续进入独立 Run 目录。历史性能整理见 [V3 资料归档](../reports/2026-10-25-v3-baseline/README.md)，不作为本页候选范围的上限。', '']
    (ROOT / 'recipes/groot-n17-5090-candidates.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'total':len(rows),**counts}))


if __name__ == '__main__':
    main()
