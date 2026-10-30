import json,sqlite3,statistics,difflib,re,csv,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';OUT=ROOT/'analysis'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
RECIPES=['nvfp4-fused','fp8-fused','fp8-unfused','nvfp4-unfused']
LABELS=['NVFP4 · Fused Backbone','FP8 · Fused Backbone','FP8 · Unfused Backbone','NVFP4 · Unfused Backbone']
def kernels(c,a,b):
 return [dict(r) for r in c.execute('SELECT k.start,k.end,k.streamId stream,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread regs,s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.start<? ORDER BY k.start',(a,b))]
def db(p):
 c=sqlite3.connect(p);c.row_factory=sqlite3.Row;return c
def stage(op):
 if op['phase']!='Backbone':return op['phase']
 if op['api'] in ['Tensor.index_copy_','Tensor.zero_']:return 'DeepStack'
 for f in op['frames']:
  fn=f['function']
  if 'deepstack' in fn:return 'DeepStack'
  if 'vit_forward' in fn:return 'ViT'
  if 'llm_forward' in fn:return 'LLM'
  if 'vlln' in fn or 'vl_self' in fn:return 'VL'
 return 'Backbone'
def describe(op):
 api=op['api'];st=stage(op);layer='';ctx={}
 for frame in reversed(op['frames']):ctx.update(frame['locals'])
 ctx={k:v for k,v in ctx.items() if not isinstance(v,int) or abs(v)<1000000}
 for frame in op['frames']:
  if 'li' in frame['locals']:layer=str(frame['locals']['li']);break
  if 'deepstack' in frame['function'] and 'j' in frame['locals']:layer=str(frame['locals']['j']);break
 # Native pybind signatures expose the actual launch argument names.
 sig=op['signature'].split('\n')[0];params=[]
 match=re.search(r'\((.*)\)\s*->',sig)
 if match:params=[p for p in re.findall(r'(\w+)\s*:',match.group(1)) if p!='self']
 args={k:v for k,v in zip(params,op['args'])};args.update(op['kwargs'])
 dimkeys={'M','N','K','m','n','k','rows','cols','dim','hidden_dim','seq_len','seqlen','num_heads','head_dim','batch','seqlen_q','seqlen_k','num_heads_q','num_heads_kv','eps','n_splits','kv_len','q_len','ld_qkv'}
 dims={k:v for k,v in args.items() if k in dimkeys and isinstance(v,(int,float)) and abs(v)<1000000}
 info='执行原生算子；参数及源码见下方'
 if 'residual_norm_quant' in api:info='合并 Bias/残差与 LayerNorm 或 RMSNorm，直接生成下一次 GEMM 的 FP8 输入；可选更新残差缓冲'
 elif 'bias_gelu_quant' in api:info='Bias → GELU → FP8 量化，在一个 Kernel 中完成'
 elif 'silu_mul_split_fp8' in api:info='SiLU(gate) × up → FP8，融合激活、逐元素乘法和量化'
 elif 'qkv_split' in api:info='读取合并 QKV 投影；合并适用的 Bias、Q/K Norm、RoPE，并写出 Attention 操作数'
 elif 'partials_attention' in api:info='直接读取 NVFP4 GEMM 的 FP32 分块结果；归约、缩放、Bias 后执行 Attention，输出 BF16'
 elif 'skinny_gemm_bf16_act' in api:info='BF16 激活在 GEMM 内量化为 NVFP4，与 NVFP4 权重相乘，输出 FP32 K 分块结果'
 elif 'skinny_gemm' in api:info='NVFP4 激活 × NVFP4 权重；自定义 skinny GEMM 沿 K 分块，输出 FP32 部分结果'
 elif 'sum_bias_gelu' in api:info='分块归约、缩放、Bias、GELU 和 NVFP4 量化融合，输出后续 GEMM 的输入及 scales'
 elif 'sum_bias_residual_norm' in api:info='分块归约、缩放、Bias、残差、(Ada)LayerNorm 融合；输出 NVFP4 或最终 BF16'
 elif 'GemmRunner.' in api:info='矩阵乘法；精度与输出类型由原生 API 名称标明，实际 M/N/K 见参数'
 elif 'quant' in api:info='读取浮点激活，按 scale 转换为低精度数据；NVFP4 还生成 block scales'
 elif 'norm' in api:info='归一化输入；AdaLN 还施加条件缩放与偏移，独立路径写出浮点中间结果'
 elif 'fwd_bf16_causal' in api:info='FlashAttention-2 causal GQA；BF16 Q/K/V → BF16 Attention 输出，保留原始 K/V 头数'
 elif 'attention_mha_causal' in api:info='FP16 causal Attention；使用已经展开到 Query 头数的 K/V'
 elif 'fwd_fp16' in api:info='FlashAttention-2 FP16 Attention'
 elif 'bias' in api:info='将 Bias 加到 GEMM 输出；具体输入输出类型见 API'
 elif 'gelu' in api or 'silu' in api:info='逐元素激活；独立实现写出中间结果'
 elif 'rope' in api:info='对 Q/K 应用旋转位置编码'
 elif 'residual' in api:info='残差相加，更新隐藏状态'
 elif 'cast' in api:info='精度转换并写出目标张量'
 elif 'expand' in api or 'repeat' in api:info='展开或复制 K/V，为后续 Attention 准备布局'
 elif 'euler' in api:info='使用当前速度预测更新 action，得到下一去噪步输入'
 io='输入输出由下方原生 API 签名及调用源码定义。'
 if all(k in dims for k in ['M','N','K']):
  m,n,k=(dims[v] for v in ['M','N','K'])
  if 'fp8_descale_fp16' in api:io=f'输入 A FP8 [{m},{k}]、权重 B FP8 [{k},{n}]、激活/权重 scale → 输出 D FP16 [{m},{n}]'
  elif 'bf16_nn' in api:io=f'输入 A BF16 [{m},{k}]、权重 B BF16 [{k},{n}] → 输出 BF16 [{m},{n}]'
  else:io=f'逻辑矩阵乘：A [{m},{k}] × W [{k},{n}] → 输出 [{m},{n}]；物理布局及精度见 API / 源码'
 elif 'fp8_residual_norm_quant' in api:
  io=f"FP16 隐藏状态、可选残差/Bias、Norm 参数 → 更新隐藏状态和 FP8 输出（或 FP16 输出）；维度 {dims}"
 elif 'fp8_qkv_split' in api:io='合并投影 QKV（FP16）→ Q、K、V；LLM 输出 BF16，ViT 输出 FP16；包含适用的 Norm / RoPE'
 elif 'fwd_bf16_causal' in api:io='BF16 Q [S,16,128]，K/V [S,8,128] → BF16 O [S,16,128]；S 见实际 seqlen 参数'
 elif 'attention_mha_causal' in api:io='FP16 Q/K/V [S,16,128]（K/V 已展开）→ FP16 O [S,16,128]'
 # Dtype/shape details remain source-backed, not inferred from kernel names alone.
 frames=[{**f,'locals':{k:v for k,v in f['locals'].items() if not isinstance(v,int) or abs(v)<1000000}} for f in op['frames']]
 return dict(api=api,stage=st,layer=layer,description=info,io=io,dimensions=dims,context=ctx,frames=frames,signature=op['signature'])
names=[];ids={}
def intern(name):
 if name not in ids:ids[name]=len(names);names.append(name)
 return ids[name]
def key(k):return (k['name'],k['gridX'],k['gridY'],k['gridZ'],k['blockX'],k['blockY'],k['blockZ'])
data=[]
zone=(R/'timezone.txt').read_text().strip()
with (R/'telemetry.csv').open() as f:
 telemetry=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
for t in telemetry:t['ns']=int(datetime.datetime.strptime(t['timestamp']+zone,'%Y/%m/%d %H:%M:%S.%f%z').timestamp()*1e9)
for recipe,label in zip(RECIPES,LABELS):
 base=R/recipe;ops=read(base/'mapping-v2-data/operations.json');mc=db(base/'mapping-v2.sqlite');mapped=[]
 for row in mc.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'OP|%' ORDER BY start").fetchall():
  idx=int(row['text'].split('|')[1]);ks=kernels(mc,row['start'],row['end'])
  for k in ks:k['op']=idx;mapped.append(k)
 tc=db(base/'trace.sqlite');requests=[]
 for row in tc.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'REQUEST|%' ORDER BY start").fetchall():
  ks=kernels(tc,row['start'],row['end']);requests.append(dict(request=row['text'],span=(max(k['end'] for k in ks)-ks[0]['start'])/1000,ks=ks))
 median=statistics.median(r['span'] for r in requests);chosen=min(requests,key=lambda r:abs(r['span']-median));ks=chosen['ks']
 matcher=difflib.SequenceMatcher(a=[key(k) for k in mapped],b=[key(k) for k in ks],autojunk=False)
 matched={}
 for a,b,size in matcher.get_matching_blocks():
  for i in range(size):matched[b+i]=mapped[a+i]['op']
 events=[];lanes={};origin=ks[0]['start'];desc=[describe(o) for o in ops]
 for idx,k in enumerate(ks):
  op=matched.get(idx);detail=desc[op] if op is not None else dict(stage='Bridge',layer='',api='未映射：边界转换或不同调度',description='保留原生 Kernel 与时间戳，未强行关联源码')
  ends=lanes.setdefault(k['stream'],[]);lane=next((i for i,e in enumerate(ends) if e<=k['start']),len(ends))
  if lane==len(ends):ends.append(k['end'])
  else:ends[lane]=k['end']
  events.append(dict(i=idx,s=(k['start']-origin)/1000,d=(k['end']-k['start'])/1000,n=intern(k['name']),op=op,stage=detail['stage'],title=detail['api'],group=detail['layer'],stream=k['stream'],lane=lane,grid=[k['gridX'],k['gridY'],k['gridZ']],block=[k['blockX'],k['blockY'],k['blockZ']],regs=k['regs'],mapping='独立 eager NVTX 调用映射，名称与 Grid/Block 序列匹配' if op is not None else '未映射'))
 wall=[read(p) for p in sorted(base.glob('wall-*/metadata.json'))]
 rounds=[w['timing']['median_ms'] for w in wall]
 samples=[t for t in telemetry if any(w['start_ns']<=t['ns']<=w['end_ns'] for w in wall)]
 gpu={k:sorted(set(t[k] for t in samples)) for k in samples[0] if k not in ['timestamp','ns']}
 stage_summary=[]
 for st in dict.fromkeys(e['stage'] for e in events):
  ee=[e for e in events if e['stage']==st];stage_summary.append(dict(stage=st,count=len(ee),sum_us=sum(e['d'] for e in ee),span_us=max(e['s']+e['d'] for e in ee)-min(e['s'] for e in ee)))
 data.append(dict(id=recipe,label=label,request=chosen['request'],span=chosen['span'],median=median,all_spans=[r['span'] for r in requests],events=events,lanes={str(k):len(v) for k,v in lanes.items()},operations=desc,mapping_coverage=len(matched)/len(ks),wall_ms=statistics.median(rounds),rounds_ms=rounds,error=wall[0]['reference_error'],calibration_hashes=wall[0]['frozen_scale_hashes'],gpu=gpu,stages=stage_summary))
 print(recipe,'wall',statistics.median(rounds),'kernels',len(ks),'mapped',len(matched),'ops',len(ops),flush=True)
docs=[];sources=[];docids={};sourceids={}
for r in data:
 for op in r['operations']:
  signature=op.pop('signature')
  if signature not in docids:docids[signature]=len(docs);docs.append(signature)
  op['doc']=docids[signature]
  frames=[]
  for frame in op['frames']:
   site={k:v for k,v in frame.items() if k!='locals'};key=json.dumps(site,sort_keys=True)
   if key not in sourceids:sourceids[key]=len(sources);sources.append(site)
   frames.append(dict(site=sourceids[key],locals=frame['locals']))
  op['frames']=frames
payload=dict(names=names,recipes=data,docs=docs,sources=sources)
(OUT/'execution-data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
summary=[{k:v for k,v in r.items() if k not in ['events','operations']} for r in data]
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
html=(OUT/'site.html').read_text(encoding='utf-8').replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
(ROOT/'flashrt_recipe_execution.html').write_text(html,encoding='utf-8')
