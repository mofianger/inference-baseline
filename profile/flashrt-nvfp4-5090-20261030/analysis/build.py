"""Join an unmodified Graph replay to separately collected source annotations."""
import json,statistics,bisect,difflib,gzip,base64,re,hashlib
from pathlib import Path
from semantics import describe
RUN=Path(__file__).resolve().parents[1]; R=RUN/'reports'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
meta=read(R/'metadata.json'); catalog=read(R/'catalog.json')
m=read(R/'mapping.json')['traceEvents']
mk=[e for e in m if e.get('cat')=='kernel']
rt={e['args'].get('correlation'):e for e in m if e.get('cat') in ('cuda_runtime','cuda_driver')}
ann=sorted([e for e in m if e.get('cat')=='user_annotation' and e['name'].startswith('MAP')],key=lambda e:e['ts']);ats=[e['ts'] for e in ann]
labels=[]
for e in mk:
 api=rt[e['args']['correlation']];v=ann[bisect.bisect_right(ats,api['ts'])-1]
 assert v['ts']<=api['ts']<=v['ts']+v['dur'];labels.append(v['name'])

def part(label):
 def idx(fn,key='li'):
  seg=next(x for x in label.split('|') if x.startswith(fn+':'));v=re.search(r'\b'+key+r'=(\d+)',seg);return int(v.group(1)) if v else -1
 if 'deepstack_merger_forward:' in label:return 'DeepStack',f'DeepStack {idx("deepstack_merger_forward","j")+1}'
 if 'qwen3vl_vit_forward:' in label:return 'ViT',f'ViT {idx("qwen3vl_vit_forward")+1:02}'
 if 'qwen3vl_llm_forward:' in label:return 'LLM',f'LLM {idx("qwen3vl_llm_forward")+1:02}'
 if 'vlln_vl_self_attn_forward:' in label:
  i=idx('vlln_vl_self_attn_forward');return 'VL','VLLN' if catalog[label]['line']==183 else f'VL {i+1:02}'
 if '_cross_kv_fwd:' in label:return 'CrossKV','Cross K/V'
 if '_state_fwd:' in label:return 'State','State encoder'
 if '_step_fwd:' in label:
  s=idx('_step_fwd','step')+1
  if catalog[label]['file'].endswith('pipeline_rtx_nvfp4.py') and catalog[label]['line']==95:return f'Step {s}',f'Step {s} / Input AdaLN'
  if 'dit_forward_nvfp4:' in label:return f'Step {s}',f'Step {s} / DiT {idx("dit_forward_nvfp4")+1:02}'
  return f'Step {s}',f'Step {s} / '+('Action encoder' if '_ae_fwd:' in label else 'Decode + Euler')
 return 'Bridge','DeepStack injection'

def trace(scope):
 events=read(R/f'{scope}.json')['traceEvents']
 ranges=sorted([e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('REQUEST|')],key=lambda e:e['ts'])
 kernels=sorted([e for e in events if e.get('cat')=='kernel'],key=lambda e:e['ts'])
 runs=[[k for k in kernels if r['ts']<=k['ts']<r['ts']+r['dur']] for r in ranges]
 assert len(runs)==10
 assert len({len(x) for x in runs})==1
 signature=lambda k:(k['name'],tuple(k['args']['grid']),tuple(k['args']['block']))
 assert all(list(map(signature,x))==list(map(signature,runs[0])) for x in runs)
 spans=[max(k['ts']+k['dur'] for k in x)-x[0]['ts'] for x in runs]
 pick=min(range(10),key=lambda i:abs(spans[i]-statistics.median(spans)));ks=runs[pick]
 return events,runs,ks,spans,pick

events,runs,ks,spans,pick=trace('frontend'); join={}
for op,a,b,c,d in difflib.SequenceMatcher(None,[k['name'] for k in ks],[k['name'] for k in mk],autojunk=False).get_opcodes():
 if op=='equal':
  for i,j in zip(range(a,b),range(c,d)):
   assert ks[i]['args']['grid']==mk[j]['args']['grid'];assert ks[i]['args']['block']==mk[j]['args']['block'];join[i]=j
 else:assert op=='delete' and all('memcpy32_post' in k['name'] or 'at::native::unrolled_elementwise' in k['name'] for k in ks[a:b])
assert len(join)==len(mk)==1575
nodes=[]
for i,k in enumerate(ks):
 label=labels[join[i]] if i in join else ''
 c=catalog[label] if label else {}
 stage,group=part(label) if label else ('Bridge','Input / output casts')
 if 'memcpy32_post' in k['name']:
  stage,group=part(labels[join[max(j for j in join if j<i)]])
 sem=describe(c,label,stage,group,k['name'],k['args']['grid'])
 if not label and 'memcpy32_post' not in k['name']:
  shape='[768,1024]' if i==0 else '[217,2048]' if i==1 else '[1,40,132]'
  sem.update(title='Patch feature cast' if i==0 else 'LLM embedding cast' if i==1 else 'Action output cast',input=shape+' BF16',output=shape+(' FP16' if i<2 else ' FP32'),precision='BF16 → '+('FP16' if i<2 else 'FP32'),note='run_backbone_graph 输入转换' if i<2 else 'frontend 返回动作的精度转换')
 if 'memcpy32_post' in k['name']:
  st=k['args']['grid'][0]==3
  sem.update(title='Copy state token' if st else 'Copy action tokens',input='[1,1536] BF16' if st else '[40,1536] BF16',output='DiT h[0,:] BF16' if st else 'DiT h[1:41,:] BF16')
 ds=[x[i]['dur'] for x in runs]
 nodes.append(dict(i=i,name=k['name'],s=round(k['ts']-ks[0]['ts'],3),d=k['dur'],median=statistics.median(ds),min=min(ds),max=max(ds),grid=k['args']['grid'],block=k['args']['block'],stream=k['args']['stream'],stage=stage,group=group,source=c.get('file',''),line=c.get('line',0),code=c.get('code',''),callstack=label,**sem))
assert all(n['input'] and n['output'] and n['title'] for n in nodes)
groups=[]
for n in nodes:
 if not groups or groups[-1]['label']!=n['group']:groups.append(dict(label=n['group'],stage=n['stage'],ids=[]))
 n['g']=len(groups)-1;groups[-1]['ids'].append(n['i'])
for g in groups:
 ids=g['ids'];g['span']=statistics.median(max(x[i]['ts']+x[i]['dur'] for i in ids)-min(x[i]['ts'] for i in ids) for x in runs)
 g['sum']=statistics.median(sum(x[i]['dur'] for i in ids) for x in runs)
stages=[]
def union(intervals):
 end=-float('inf');total=0
 for a,b in sorted(intervals):
  total+=max(0,b-max(a,end));end=max(end,b)
 return total
for stage in dict.fromkeys(n['stage'] for n in nodes):
 ids=[n['i'] for n in nodes if n['stage']==stage]
 chunks=[g['ids'] for g in groups if g['stage']==stage]
 stages.append(dict(stage=stage,ids=ids,span=statistics.median(union([(min(x[i]['ts'] for i in chunk),max(x[i]['ts']+x[i]['dur'] for i in chunk)) for chunk in chunks]) for x in runs),sum=statistics.median(sum(x[i]['dur'] for i in ids) for x in runs)))
# Bridge casts are separated in time, so do not describe their bounding box as a stage cost.
for s in stages:
 if s['stage']=='Bridge':s['span']=None
ie,ir,ik,isp,ipick=trace('integrated')
assert [k['name'] for k in ik[-len(nodes):]]==[n['name'] for n in nodes]
integrated=[dict(i=i,name=k['name'],s=round(k['ts']-ik[0]['ts'],3),d=k['dur'],median=statistics.median(x[i]['dur'] for x in ir),stage='Official support' if i<len(ik)-len(nodes) else nodes[i-(len(ik)-len(nodes))]['stage']) for i,k in enumerate(ik)]
summary=dict(metadata=meta,kernel_count=len(nodes),mapped_compute=len(join),group_count=len(groups),selected_replay=pick+1,gpu_span_us=statistics.median(spans),gpu_spans_us=spans,integrated_gpu_span_us=statistics.median(isp),integrated_kernel_count=len(ik),integrated_replay=ipick+1,stage_stats=stages,validation='All 1575 compute kernels matched by full name/order/grid/block; 10 replay signatures stable. Eleven casts/internal copy kernels separately identified.',hashes={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['frontend.json','integrated.json','mapping.json','catalog.json']})
data=dict(summary=summary,nodes=nodes,groups=groups,integrated=integrated)
(RUN/'analysis/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(RUN/'analysis/kernels.json').write_text(json.dumps(nodes,ensure_ascii=False,indent=2),encoding='utf-8')
packed=base64.b64encode(gzip.compress(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode())).decode()
template=(RUN/'harness/site.html').read_text(encoding='utf-8')
(RUN/'flashrt_nvfp4_profile.html').write_text(template.replace('__DATA__',packed),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('metadata','hashes','stage_stats')},indent=2))
