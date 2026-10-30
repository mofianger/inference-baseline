"""Logical tensor contracts audited against pinned FlashRT source, not hardware counters.
Shapes omit batch=1. Source expressions remain attached to every node for audit.
"""
import re
def describe(c,label,stage,group,name,grid):
 line=c.get('line');file=c.get('file','').split('/')[-1];v=c.get('scalars',{})
 def result(title,inp,out,fusion='',precision='BF16',kind='Pointwise',note=''):
  return dict(title=title,input=inp,output=out,fusion=fusion or '单项操作',precision=precision,kind=kind,note=note,pdl=file=='pipeline_rtx_nvfp4.py')
 def gemm(title,m,n,k,precision='FP8',output='FP16',fusion='矩阵乘法与量化尺度恢复'):
  return result(title,f'激活 [{m}, {k}]；权重 [{k}, {n}]（逻辑转置布局）',f'[{m}, {n}] {output}',fusion,precision,'GEMM',f'(M,N,K)=({m},{n},{k})；权重/激活 scale 随输入传入。')
 if not c:
  if 'memcpy32_post' in name:return result('拼接状态 / 动作 token','状态 [1,1536] 或动作 [40,1536] BF16','DiT h [41,1536] 的对应行','设备缓冲区拷贝','BF16','Copy')
  return result('Frontend 精度转换','输入特征或输出动作张量','相同形状、目标 dtype 的张量','精度转换','FP16 / BF16 / FP32','Copy','图外转换；边界输入为准备好的视觉/文字特征，输出为 [1,40,132] FP32。')
 if file=='pipeline_rtx_nvfp4.py':
  if line==92:
   n,k=v['n'],v['k'];split=k//(256 if n==1536 and k==1536 else 512)
   title=next((s.split('what=')[1].split(',')[0] for s in label.split('|') if 'what=' in s),'NVFP4 GEMM')
   return result(title,f'激活 [41,{k}] NVFP4 + 每16元素scale；权重 [{n},{k}] NVFP4 + scale',f'partials [{split},41,{n}] FP32','Split-K 矩阵乘法；PDL 前置权重加载','NVFP4 → FP32','GEMM',f'(M,N,K)=(41,{n},{k})；K 分成 {split} 段，每段结果尚未求和。')
  if line==121:return result('O projection','Attention O [41,1536] BF16；W [1536,1536] NVFP4','partials [6,41,1536] FP32','加载激活时量化为 NVFP4 + Split-K GEMM','BF16 → NVFP4 → FP32','GEMM','(M,N,K)=(41,1536,1536)；6 份部分和。')
  if line in (106,113):
   self_=line==106;kv=41 if self_ else v['kv']
   inp='QKV partials [3,41,4608] FP32' if self_ else f'Q partials [6,41,1536] FP32；缓存 K/V 各 [{kv},32,48] BF16（行跨度24576）'
   return result('Self-Attention' if self_ else ('Cross-Attention / text' if kv==25 else 'Cross-Attention / image'),inp,'O [41,32,48] BF16','合并投影 partials + scale + Bias + QKᵀ + 在线 Softmax + PV','BF16 / FP32 accumulation','Attention','32 heads × 3 query tiles = 96 CTA；每 CTA 最多16个 query。Cross K/V 可在 PDL wait 前加载。')
  if line==95:return result('首层 AdaLN + quant','h [41,1536] BF16；时间调制 scale/shift [1536]','激活 [41,1536] NVFP4 + scales [41,96]','LayerNorm + 时间调制 + 动态分块量化','BF16 → NVFP4','Norm')
  if line==129:return result('FFN GELU + quant','partials [3,41,6144] FP32；Bias [6144]','FFN 激活 [41,6144] NVFP4 + scales [41,384]','部分和合并 + scale + Bias + tanh-GELU + 动态量化','FP32 → NVFP4','Activation')
  if line in (123,136,140):
   split=6 if line==123 else 12
   return result('O residual + LN + quant' if line==123 else 'FFN residual + output AdaLN' if line==140 else 'FFN residual + next AdaLN + quant',f'partials [{split},41,1536] FP32；h [41,1536] BF16；Bias/调制向量','更新 h [41,1536] BF16；'+('hmod [41,1536] BF16' if line==140 else '下一投影输入 [41,1536] NVFP4 + scales'),'部分和合并 + scale + Bias + 残差 + '+('LayerNorm' if line==123 else 'AdaLN')+('' if line==140 else ' + NVFP4 量化'),'FP32 / BF16 / NVFP4','Norm','跨算子/跨层融合，保留 BF16 残差支路。')
 if file=='pipeline_rtx_fp8_fused.py':
  m,d,ff=(768,1024,4096) if stage=='ViT' else (192,4096,4096) if stage=='DeepStack' else (217,2048,6144) if stage=='LLM' else (217,2048,8192)
  gs={59:('QKV',768,3072,1024),67:('O projection',768,1024,1024),72:('FFN Up',768,4096,1024),76:('FFN Down',768,1024,4096),99:('Merger FC1',192,4096,4096),103:('Merger FC2',192,2048,4096),139:('QKV',217,4096,2048),151:('O projection',217,2048,2048),155:('Gate projection',217,6144,2048),157:('Up projection',217,6144,2048),160:('Down projection',217,2048,6144),194:('Q / K / V projection',217,2048,2048),201:('O projection',217,2048,2048),206:('FFN Up',217,8192,2048),210:('FFN Down',217,2048,8192)}
  if line in gs:return gemm(*gs[line])
  if line in (56,69,96,137,153,190,203):return result('Residual + Norm + FP8 quant',f'h [{m},{d}] FP16；待合并投影、Bias/Norm参数；LLM 前三层还接入 DeepStack',f'归一化激活 [{m},{d}] FP8；残差 h 原位更新','按调用参数合并残差/Bias/DeepStack + LayerNorm或RMSNorm + 静态量化','FP16 → FP8','Norm','首层可能没有待合并残差；精确操作开关与缓冲区见源码调用。')
  if line==183:return result('VLLN','LLM h 与末层待加残差 [217,2048] FP16','VL h [217,2048] FP16','残差相加 + VLLN','FP16','Norm')
  if line==30:return result('Residual writeback',f'h 与投影增量 [{m},{d}] FP16',f'更新后的 h [{m},{d}] FP16','残差 + 可选 Bias / DeepStack；按调用需要写出','FP16','Pointwise')
  if line in (62,141,196):
   out='Q [217,16,128]；K/V [217,8,128] BF16' if line==141 else f'Q/K/V 各 [{m},{16 if stage=="ViT" else 32},64] FP16'
   return result('QKV split + position',f'投影输出 [{m},{4096 if line==141 else 3072 if line==62 else 2048}] FP16；位置/Norm/Bias参数',out,'QKV布局整理 + '+('Q/K RMSNorm + M-RoPE + BF16转换' if line==141 else 'Bias + RoPE' if line==62 else 'Bias'),'FP16 / BF16','Position')
  if line==144:return result('Causal GQA / FA2','Q [217,16,128]；K/V [217,8,128] BF16','O [217,16,128] BF16','QKᵀ + causal mask + Softmax + PV；原生GQA','BF16','Attention')
  if line in (66,150,200):return result('Attention output quant',f'O [{m},{d}] '+('BF16' if line==150 else 'FP16'),f'激活 [{m},{d}] FP8','静态scale量化','FP16 / BF16 → FP8','Quantize')
  if line in (74,101,208):return result('Bias + GELU + quant',f'FC1 [{m},{ff}] FP16；Bias [{ff}]',f'[{m},{ff}] FP8','Bias + tanh-GELU + 静态量化','FP16 → FP8','Activation')
  if line==105:return result('Merger Bias','Merger FC2 [192,2048] FP16；Bias [2048]','DeepStack [192,2048] FP16','Bias相加','FP16')
  if line==159:return result('SwiGLU + quant','Gate / Up 各 [217,6144] FP16','激活 [217,6144] FP8','SiLU(Gate) × Up + 静态量化','FP16 → FP8','Activation')
 if file=='attn_backend_groot_n17_backbone.py':
  shape='[3,256,16,64]' if stage=='ViT' else '[1,217,32,64]'
  return result('Multi-view FA2' if stage=='ViT' else 'VL FA2',f'Q/K/V {shape} FP16',f'O {shape} FP16','QKᵀ + Softmax + PV','FP16','Attention')
 if file=='groot_n17_rtx_fp8.py':return result('DeepStack scatter','DeepStack [192,2048] FP16；视觉位置索引 [192]','注入缓冲区 [217,2048] FP16','按视觉位置写入特征；文字位置保持零','FP16','Copy')
 if file=='groot_n17_rtx_fp4.py':
  rows=25 if 'name=text' in label else 192
  if line==503:return result('Backbone output cast','[217,2048] FP16','[217,2048] BF16','精度转换','FP16 → BF16','Copy')
  if line in (504,506):return result('Gather text' if line==504 else 'Gather image','[217,2048] BF16；位置索引',f'[{25 if line==504 else 192},2048] BF16','按文本或视觉位置收集','BF16','Copy')
  if line==509:return result('Cross K/V activation quant',f'[{rows},2048] BF16；固定全局scale',f'[{rows},2048] NVFP4 + 每16元素动态scale','动态分块量化 + scale布局写入','BF16 → NVFP4','Quantize')
  if line==512:return gemm('Cross K/V / '+('text' if rows==25 else 'image'),rows,24576,2048,'NVFP4','BF16','8层 × K/V 投影合并为一个GEMM + Bias；各层通过列切片读取')
 if file=='groot_n17_thor.py':
  gs={784:('State FC1',1,1024,132),788:('State FC2',1,1536,1024),796:('Action FC1',40,1536,132),801:('Action FC2',40,1536,3072),807:('Action FC3',40,1536,1536),823:('Output projection',41,1024,1536),826:('Decoder FC1',40,1024,1024),830:('Decoder FC2',40,132,1024)}
  if line in gs:return gemm(*gs[line],precision='BF16',output='BF16',fusion='矩阵乘法；Bias在后续Kernel处理')
  shapes={786:(1,1024),787:(1,1024),790:(1,1536),798:(40,1536),803:(40,1536),804:(40,1536),805:(40,1536),806:(40,1536),809:(40,1536),810:(40,1536),825:(41,1024),828:(40,1024),829:(40,1024),832:(40,132)}
  if line in shapes:
   shape=str(list(shapes[line]));cast=line in (804,806);act=line in (787,805,829);title='BF16 → FP16' if line==804 else 'FP16 → BF16' if line==806 else 'SiLU' if line==805 else 'ReLU' if act else 'Add position embedding' if line==810 else 'Add Bias'
   return result(title,shape+' '+('FP16' if line in (805,806) else 'BF16')+('；位置/Bias向量' if not cast and not act else ''),shape+' '+('FP16' if line in (804,805) else 'BF16'),title,'FP16 / BF16','Copy' if cast else 'Activation' if act else 'Pointwise')
  if line==799:return result('Concat timestep','动作特征 [40,1536]；时间特征 [40,1536]','[40,3072] BF16','拼接','BF16','Copy')
  if line==833:return result('Euler update','动作 x [40,132]；速度 v [40,132]；步长 dt','下一步动作 x [40,132] BF16','x ← x + dt × v，原位更新','BF16','Pointwise')
 raise ValueError(f'Unmapped source contract: {file}:{line} {group}')
