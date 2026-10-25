"""Build verified kernel/source/CTA joins; never synthesize SM placement."""
import json,bisect,difflib,statistics,math,struct,collections,gzip,base64,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];R=RUN/'reports';A=RUN/'analysis'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
t=read(R/'trace.json')['traceEvents'];ranges=sorted([x for x in t if x.get('cat')=='user_annotation' and x['name'].startswith('FLASHRT_REQUEST')],key=lambda x:x['ts']);kernels=[x for x in t if x.get('cat')=='kernel'];runs=[]
for r in ranges:
 k=sorted([x for x in kernels if r['ts']<=x['ts']<r['ts']+r['dur']],key=lambda x:x['ts']);runs.append(k)
assert all(len(k)==1586 for k in runs)
spans=[max(x['ts']+x['dur'] for x in k)-k[0]['ts'] for k in runs];pick=min(range(len(runs)),key=lambda i:abs(spans[i]-statistics.median(spans)));selected=runs[pick]
assert all([x['name'] for x in k]==[x['name'] for x in selected] for k in runs)
m=read(R/'mapping.json')['traceEvents'];mk=[x for x in m if x.get('cat')=='kernel'];rt={x['args'].get('correlation'):x for x in m if x.get('cat') in ('cuda_runtime','cuda_driver')};ann=sorted([x for x in m if x.get('cat')=='user_annotation'],key=lambda x:x['ts']);starts=[x['ts'] for x in ann];labels=[]
for x in mk:
 api=rt[x['args']['correlation']];i=bisect.bisect_right(starts,api['ts'])-1;label=ann[i];assert label['ts']<=api['ts']<=label['ts']+label['dur'];labels.append(label['name'])
join={}
for op,a,b,c,d in difflib.SequenceMatcher(None,[x['name'] for x in selected],[x['name'] for x in mk],autojunk=False).get_opcodes():
 if op=='equal':
  for i,j in zip(range(a,b),range(c,d)):assert selected[i]['args']['grid']==mk[j]['args']['grid'];join[i]=j
 else:assert op=='delete' and all('memcpy32_post' in x['name'] or 'at::native::unrolled_elementwise' in x['name'] for x in selected[a:b])
assert len(join)==1575
import re
def part(label):
 def value(fn,key='li'):
  seg=next(x for x in label.split('|') if x.startswith(fn+':'));v=re.search(r'\b'+key+r'=(\d+)',seg);return int(v.group(1)) if v else -1
 if 'deepstack_merger_forward:' in label:return 'DeepStack',f'DeepStack {value("deepstack_merger_forward","j")+1}'
 if 'qwen3vl_vit_forward:' in label:return 'ViT',f'ViT {value("qwen3vl_vit_forward")+1:02}'
 if 'qwen3vl_llm_forward:' in label:return 'LLM',f'LLM {value("qwen3vl_llm_forward")+1:02}'
 if 'vlln_vl_self_attn_forward:' in label:return 'VL',f'VL {value("vlln_vl_self_attn_forward")+1:02}'
 if '_cross_kv_fwd:' in label:return 'CrossKV','Cross K/V'
 if '_state_fwd:' in label:return 'State','State encoder'
 if '_step_fwd:' in label:
  s=value('_step_fwd','step')+1
  if 'dit_forward_nvfp4:' in label:return f'Step {s}',f'Step {s} / DiT {value("dit_forward_nvfp4")+1:02}'
  return f'Step {s}',f'Step {s} / '+('Action encoder' if '_ae_fwd:' in label else 'Decode + Euler')
 return 'Bridge','DeepStack injection'
def family(n):
 if 'gemm_kernel' in n or 'nvjet' in n or 'gemvx' in n or 'cutlass::Kernel' in n:return 'GEMM'
 if 'attention' in n or 'flash_fwd' in n:return 'Attention'
 if 'norm' in n:return 'Norm + fusion'
 if 'gelu' in n or 'silu' in n or 'relu' in n:return 'Activation + fusion'
 if 'quant' in n:return 'Quantization'
 if 'copy' in n or 'memcpy' in n or 'cast' in n:return 'Copy / cast'
 return 'Other'
nodes=[];offset=selected[0]['ts']
for i,x in enumerate(selected):
 label=labels[join[i]] if i in join else 'Graph copy' if 'memcpy32_post' in x['name'] else 'Frontend cast'
 stage,group=part(label) if i in join else ('Bridge',label)
 if label=='Graph copy':stage,group=part(labels[join[max(j for j in join if j<i)]])
 nodes.append(dict(i=i,name=x['name'],s=round(x['ts']-offset,3),d=x['dur'],median=statistics.median(k[i]['dur'] for k in runs),grid=x['args']['grid'],block=x['args']['block'],stream=x['args']['stream'],stage=stage,group=group,source=label,family=family(x['name']),cta=[]))
# CTA recorder contains one launch per mapped compute kernel in the same order.
names=[line.split('\t',1)[1].strip() for line in (R/'cta.names.tsv').read_text().splitlines()];assert len(names)==len(mk)
assert names==[x['name'] for x in mk]
inverse={v:k for k,v in join.items()};records=list(struct.iter_unpack('<IIQQIIII',(R/'cta.bin').read_bytes()));origin=min(x[2] for x in records)
for launch,cta,start,end,sm,entries,exits,pad in records:
 assert entries==exits and end>=start and 0<=sm<170
 node=nodes[inverse[launch]];assert entries==math.prod(node['block']);assert pad==math.prod(node['grid'])
 node['cta'].append([sm,cta,round((start-origin)/1000,3),round((end-start)/1000,3)])
shapes=[list(map(int,line.split())) for line in (R/'cta.capture.tsv').read_text().splitlines()]
assert len(shapes)==len(mk)
changes=[]
for i,j in join.items():
 nodes[i]['cta_grid']=shapes[j][:3]
 assert nodes[i]['block']==shapes[j][3:]
 nodes[i]['cta_expected']=math.prod(shapes[j][:3])
 nodes[i]['cta_complete']=len(nodes[i]['cta'])==nodes[i]['cta_expected']
 if not nodes[i]['cta_complete']:changes.append({'node':i,'expected':nodes[i]['cta_expected'],'recorded':len(nodes[i]['cta'])})
 if nodes[i]['grid']!=shapes[j][:3]:changes.append({'node':i,'original':nodes[i]['grid'],'instrumented':shapes[j][:3]})
groups=[]
for n in nodes:
 if not groups or groups[-1]['label']!=n['group']:groups.append({'label':n['group'],'stage':n['stage'],'ids':[]})
 n['g']=len(groups)-1;groups[-1]['ids'].append(n['i'])
for g in groups:
 ids=g['ids'];g['sum_us']=statistics.median(sum(k[i]['dur'] for i in ids) for k in runs);g['span_us']=statistics.median(max(k[i]['ts']+k[i]['dur'] for i in ids)-min(k[i]['ts'] for i in ids) for k in runs)
agg=[]
for stage in dict.fromkeys(n['stage'] for n in nodes):
 ids=[n['i'] for n in nodes if n['stage']==stage];agg.append({'stage':stage,'nodes':len(ids),'sum_us':statistics.median(sum(k[i]['dur'] for i in ids) for k in runs)})
summary={'selected_replay':pick,'gpu_spans_us':spans,'gpu_span_median_us':statistics.median(spans),'kernel_count':len(nodes),'cta_count':len(records),'cta_kernel_count':len(join),'sm_count':170,'validation':'compute names/order and block dimensions matched; all recorded CTA thread entry/exit counts matched; see coverage exceptions','coverage_exceptions':changes,'stages':agg,'baseline':read(A/'timeline-metadata.json'),'instrumented':read(A/'cta-metadata.json'),'nsys':read(A/'nsys-summary.json'),'sources':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['trace.json','mapping.json','cta.bin','cta.names.tsv','cta.capture.tsv']}}
(A/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
payload={'summary':summary,'nodes':nodes,'groups':groups};packed=base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode())).decode();template=(RUN/'harness/report.html').read_text(encoding='utf-8-sig');(RUN/'flashrt_nvfp4_profile.html').write_text(template.replace('__PAYLOAD__',packed),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k not in ['baseline','instrumented','sources']},indent=2));print('HTML bytes', (RUN/'flashrt_nvfp4_profile.html').stat().st_size)
