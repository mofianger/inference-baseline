"""Build a standalone, offline reader-facing report from recorded measurements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'reports/2026-10-25-5090-groot-new-recipes'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
# Titles and explanations describe the tested implementation, not a generic precision label.
RECIPES = {
 'flashrt-nvfp4': ('FlashRT', 'FP8 骨干 + NVFP4 动作生成', '看图、读指令主要用 FP8；生成动作的主要矩阵运算改用 NVFP4。把多项计算合并，并用 CUDA Graph 减少 CPU 逐项提交的开销。', '同一条请求的 4 步动作生成，共用已经算好的图像和指令信息；新请求仍重新计算。注意力等部分运算保留较高精度，并非全模型都是 4 位。'),
 'flashrt-fp8': ('FlashRT', '融合 FP8', '看图、读指令主要用 FP8；动作生成中的部分矩阵运算也用 FP8，其余保留 BF16。把相邻计算合并，减少中间数据读写，再用 CUDA Graph 重放执行步骤。', '这是 FP8 与 BF16 混合的路径，不是把整个模型统一改成 FP8。每次请求都使用当前图像、状态和初始噪声。'),
 'embodiinfer-bf16': ('EmbodiInfer', 'BF16 · 编译 + 两段 CUDA Graph', '保留 BF16 等原有数值格式。先把看图、读指令和处理状态的过程录成一段执行计划，再把完整 4 步动作生成录成另一段；同时用 Inductor 编译、合并动作生成中的计算。', '“前缀图”就是前一段，“完整去噪图”就是后一段。复用的是执行步骤，不是上一帧的结果；4 步之间不变的条件信息只在本次请求内复用。'),
 'phyai-bf16': ('PhyAI', 'BF16 · CUDA Graph', '用 BF16 等原有数值格式运行模型，把骨干和动作生成的执行步骤记录成 CUDA Graph，减少每次调用时的 CPU 调度。', '这是后续 PhyAI 量化配方的对照。图像理解与动作生成都参与计时，没有把前半段模型移到计时外。'),
 'reference': ('NVIDIA', '官方 PyTorch BF16 参考', '按官方模型逐步执行，看图、读指令，再完成 4 步动作生成。不使用本页其他配方的 TensorRT 引擎或专用完整流程优化。', '它同时提供本次误差比较的参考输出。自身误差为 0 是因为和自身比较，不代表真实机器人动作一定正确。'),
 'groot-trt-baseline': ('NVIDIA', 'TensorRT · 基础执行', '把看图、读指令和生成动作等模块提前编译为 TensorRT 引擎，推理时按顺序调用。没有额外低位量化：视觉网络为 FP32，其余主要为 BF16。', 'TensorRT 会选择计算方式并合并适合合并的操作；模块之间仍由运行程序衔接。FP32 网络格式不表示所有底层乘法都强制使用同一种 FP32 指令。'),
 'groot-trt-optimized': ('NVIDIA', 'TensorRT · 优化执行', '在未量化的 TensorRT 基础上，复用同一请求内不变的条件信息，并把动作生成流程录成 CUDA Graph，减少重复工作和 CPU 调度。', '主要改变执行方式，而不是降低模型数值位数。与“基础执行”分开列出，可以看到执行组织本身的影响。'),
 'groot-trt-fp8': ('NVIDIA', 'TensorRT · FP8 混合精度', '按官方配方把视觉、语言和动作生成中的主要计算量化为 FP8，再交给 TensorRT 编译。运行时使用优化后的动作生成流程和 CUDA Graph。', '辅助模块使用 FP16，部分操作按上游策略保留较高精度。量化前用 8 条独立合成样本确定数值范围，不使用本页的评估样本。'),
 'groot-trt-nvfp4': ('NVIDIA', 'TensorRT · NVFP4 混合精度', '视觉部分用 FP8；语言部分主要用 NVFP4，但部分输出投影保留 FP8；动作生成只把前馈层改为 NVFP4，注意力部分用 FP8。再用 TensorRT 和 CUDA Graph 执行。', '辅助模块为 FP16。这条配方与 FlashRT 的 4 位分配范围不同，因此都叫 NVFP4，也不能视为同一套计算。'),
 'groot-trt-compile': ('NVIDIA', 'PyTorch · 编译动作网络', '仍使用原来的 BF16 模型，只将动作生成网络交给 torch.compile。编译器尝试合并运算并选择较快的实现；看图、读指令等部分仍由 PyTorch 执行。', '编译使用上游 max-autotune 配置。编译时间不计入延迟，但每次请求的完整模型计算仍计入。'),
 'foldquant-w8a8': ('FoldQuant', '原生 INT8 · W8A8', '先调整数值分布，减少少数特别大的数对量化的影响，再把语言和动作网络的主要矩阵运算改为 INT8。使用 FoldQuant 的专用 INT8 插件，在 TensorRT 中执行完整模型。', '语言网络使用 SR 配方，动作网络使用 SH 配方，包含不同的数值变换；共记录 304 个量化投影。视觉、编码器等其他部分保留上游浮点策略。'),
 'foldquant-awq': ('FoldQuant', 'AWQ · INT4 权重 / 16 位中间值', '用 AWQ 校准，调整权重的缩放方式，尽量减小压到 4 位带来的误差。语言和动作网络的主要权重用 INT4 保存，中间计算值保留 16 位，再导出给 TensorRT。', '这是 W4A16，不是 NVFP4，也不是权重和中间值都用 4 位。权重更省空间，并不保证在这组输入下更快。'),
 'foldquant-smoothquant': ('FoldQuant', 'SmoothQuant · INT8', '先调整权重和中间数值的比例，让少数过大的中间值更容易用 8 位表示，再把语言和动作网络的主要矩阵运算量化为 INT8，交给 TensorRT 执行。', '这是上游 ModelOpt 的 SmoothQuant 路径，与 FoldQuant 自己的 INT8 插件配方不同。校准后的权重和缩放参数一起用于导出。'),
 'flashrt-old': ('FlashRT', '早期 FP8 版本', '早期接入的 FP8 路径。每次请求仍重新计算图像和文本的嵌入并传入模型；源码版本、构建和执行路径与后来的融合 FP8 配方不同。', '作为历史记录保留。新旧结果的差异不能全部归因于某一项融合或某一个算子。'),
 'embodiinfer-old': ('EmbodiInfer', 'BF16 · 早期基础接入', '使用引擎的原生推理路径，但没有启用前缀图和完整 4 步去噪图，因此仍有较多逐步调度。', '这是早期接入结果，不能代表已经启用完整优化后的 EmbodiInfer。'),
}
phyai = {
 'fp8-a16': ('FP8 权重 / BF16 中间值', '把语言网络的权重压到 FP8，中间计算值保留 BF16。'),
 'nvfp4-a16': ('NVFP4 权重 / BF16 中间值', '把语言网络的权重按小组压到 NVFP4，中间计算值保留 BF16。'),
 'int4-a16': ('INT4 权重 / BF16 中间值', '把语言网络的权重压到 INT4，中间计算值保留 BF16。'),
 'fp8-a8': ('FP8 权重与中间值', '语言网络的权重和中间值都用 FP8；中间值的缩放范围在运行时计算。'),
 'fp8-block-a8': ('分块 FP8 权重与中间值', '权重按 128×128 的小块分别缩放，再用 FP8 表示；中间值也动态量化为 FP8。'),
 'int8-a8': ('INT8 权重与中间值', '语言网络的权重和中间值都用 INT8；中间值在运行时量化。'),
 'int4-fp8-a8': ('INT4 权重 / FP8 中间值', '语言网络的权重用 INT4，中间值用 FP8，分别减少权重存储和计算中的数值位数。'),
}
for suffix, (title, explanation) in phyai.items():
    for eager in ([False, True] if suffix in ('fp8-a8','fp8-block-a8','int8-a8','int4-fp8-a8') else [False]):
        key = 'phyai-llm-' + suffix + ('-eager' if eager else '')
        RECIPES[key] = ('PhyAI', title + (' · 关闭图' if eager else ' · CUDA Graph'),
            explanation + (' 关闭 CUDA Graph，按普通逐步调用方式运行，用来检查问题是否与图重放有关。' if eager else ' 使用现成的 Humming 计算库，并保留 CUDA Graph 执行。'),
            '只量化语言网络的 64 个融合投影层；看图和动作生成仍为原来的 BF16 等格式。通过引擎现有配置选择计算方式，没有为它新增量化实现。')
RECIPES['flashrt-fp8-fp16-head'] = ('FlashRT', 'FP8 骨干 + FP16 动作生成',
    '看图、读指令用 FP8，动作生成使用现成的 FP16 前端，作为不同精度组合的对照。',
    '部分动作网络权重先按 FP8 装载再还原为 FP16，所以它不是原权重的纯 FP16 参考。本次更换图像后输出没有变化，未纳入有效延迟。')

def measured(key, path, history=False):
    d = read(path)
    assert d['dataset_manifest_sha256'] == '562a86fc238ca10adcc2b623be09bf796b74f1403fbdd5ab6f6d1785335c2297'
    engine, title, method, note = RECIPES[key]
    unique = {x['sample_id']: x for x in d['correctness']}
    return dict(id=key, engine=engine, title=title, method=method, note=note,
        mean=d['latency']['mean_ms'], p95=d['latency']['p95_ms'],
        error=max(x['max_abs'] for x in unique.values()), mae=sum(x['mae'] for x in unique.values()) / len(unique),
        source=path.relative_to(ROOT).as_posix(), commit=d['engine_source']['commit'],
        date=d['timestamp_utc'][:10], history=history, diagnostic=key.endswith('-eager'), status='measured')

rows = [measured(x['recipe'], NEW/x['source']) for x in read(NEW/'run-index.json') if x['status']=='measured']
for key, filename in [('flashrt-nvfp4','flashrt_nvfp4'),('flashrt-fp8','flashrt_fp8'),('embodiinfer-bf16','embodiinfer_optimized')]:
    rows.append(measured(key, ROOT/f'reports/2026-10-23-5090-optimized/{filename}.json'))
for key, filename, history in [('phyai-bf16','phyai',False),('reference','reference',False),('flashrt-old','flashrt',True),('embodiinfer-old','embodiinfer',True)]:
    rows.append(measured(key, ROOT/f'reports/2026-10-23-5090-smoke/{filename}.json',history))
failures = {}
for x in read(NEW/'run-index.json'):
    if x['status'] == 'failed':
        failures[x['recipe']] = x
for key, record in failures.items():
    engine,title,method,note=RECIPES[key]
    rows.append(dict(id=key,engine=engine,title=title,method=method,note=note,status='failed',
        history=False,diagnostic=key.endswith('-eager'),source=(NEW/record['source']).relative_to(ROOT).as_posix(),
        reason='更换图像后，输出没有变化。可能复用了旧的条件信息。' if engine=='FlashRT' else '先输入 A，再输入 B，再输入 A，前后两次 A 的输出差异超过现有重复性检查的容差。'))
assert len(rows)==27 and len(failures)==8
payload = json.dumps(rows,ensure_ascii=False).replace('</','<\/')
template=(ROOT/'scripts/templates/groot_results.html').read_text(encoding='utf-8')
target=ROOT/'groot_n17_5090_results.html'
target.write_text(template.replace('__DATA__',payload),encoding='utf-8')
print(f'Built {target.name}: 17 current + 2 historical measured recipes, 8 failed recipes')
