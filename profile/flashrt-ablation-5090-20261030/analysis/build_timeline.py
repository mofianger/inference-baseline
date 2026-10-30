"""Build an offline timeline from actual Nsys requests, never median-spliced kernels."""
import json, sqlite3, statistics, difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
reference=[k for k in read(ROOT.parent/'flashrt-nvfp4-5090-20261030/analysis/kernels.json') if 'memcpy32_post' not in k['name']]
names=[];name_ids={}
def intern(s):
 if s not in name_ids:name_ids[s]=len(names);names.append(s)
 return name_ids[s]
def collect(file,prefix):
 c=sqlite3.connect(ROOT/'reports'/file);c.row_factory=sqlite3.Row
 out=[]
 for r in c.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE ? ORDER BY start',(prefix+'%',)).fetchall():
  kk=[dict(k) for k in c.execute('SELECT k.start,k.end,k.streamId,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread,s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.start<? ORDER BY k.start',(r['start'],r['end']))]
  out.append(dict(label=r['text'],kernels=kk))
 return out
off=collect('pdl/trace-0.sqlite','PDL|');on=collect('pdl/trace-1.sqlite','PDL|')
fu=collect('fusion-trace.sqlite','FUSION|')
fused=[r for r in fu if '|fused|' in r['label']];unfused=[r for r in fu if '|unfused|' in r['label']]
for r in off+on+fused:
 assert len(r['kernels'])==len(reference)
 for k,e in zip(r['kernels'],reference):
  assert [k['gridX'],k['gridY'],k['gridZ']]==e['grid']
  assert [k['blockX'],k['blockY'],k['blockZ']]==e['block']
  k.update(stage=e['stage'],title=e['title'],group=e['group'],mapping='原路径顺序与 Grid/Block 核对')
# Unchanged kernels anchor the expanded sequence. Label replacement intervals only
# when their source interval belongs to one stage; keep ambiguous boundaries explicit.
for r in unfused:
 a=fused[0]['kernels'];b=r['kernels']
 matcher=difflib.SequenceMatcher(a=[k['name'] for k in a],b=[k['name'] for k in b],autojunk=False)
 for tag,i,j,u,v in matcher.get_opcodes():
  if tag=='equal':
   for old,new in zip(a[i:j],b[u:v]):new.update(stage=old['stage'],title=old['title'],group=old['group'],mapping='未改变的 Kernel 名称序列匹配')
  else:
   stages=set(k['stage'] for k in a[i:j])
   if not stages and i and i<len(a) and a[i-1]['stage']==a[i]['stage']:stages={a[i]['stage']}
   stage=next(iter(stages)) if len(stages)==1 else 'Transition'
   for k in b[u:v]:k.update(stage=stage,title='拆分后的计算',group='Unfused control',mapping='相邻原始 Kernel 锚点之间的替换区间' if stage!='Transition' else '跨阶段替换区间，未强行归类')
 assert all('stage' in k for k in b)
def span(ks):return (max(k['end'] for k in ks)-min(k['start'] for k in ks))/1000
def pack(r,dit=False):
 ks=[k for k in r['kernels'] if not dit or k['stage'].startswith('Step')]
 base=min(k['start'] for k in ks)
 # Each stream gets separate overlap sublanes. Preserve actual overlaps verbatim.
 lanes={};events=[]
 for idx,k in enumerate(r['kernels']):
  if dit and not k['stage'].startswith('Step'):continue
  ends=lanes.setdefault(k['streamId'],[])
  lane=next((i for i,e in enumerate(ends) if e<=k['start']),len(ends))
  if lane==len(ends):ends.append(k['end'])
  else:ends[lane]=k['end']
  events.append(dict(i=idx,s=round((k['start']-base)/1000,3),d=round((k['end']-k['start'])/1000,3),n=intern(k['name']),stage=k['stage'],title=k['title'],group=k['group'],mapping=k['mapping'],stream=k['streamId'],lane=lane,grid=[k['gridX'],k['gridY'],k['gridZ']],block=[k['blockX'],k['blockY'],k['blockZ']],regs=k['registersPerThread']))
 return dict(request=r['label'],span=span(ks),events=events,lanes={str(k):len(v) for k,v in lanes.items()})
def representative(rr,dit=False):
 allr=[pack(r,dit) for r in rr];median=statistics.median(r['span'] for r in allr)
 selected=min(allr,key=lambda r:abs(r['span']-median))
 selected.update(median=median,all_spans=[r['span'] for r in allr],selection='所有已采集请求中，GPU 跨度最接近中位数的一次真实请求')
 return selected
data=dict(names=names,panels=[
 dict(id='standard',title='标准路径 · PDL + Fusion',scope='完整 aux → action',subtitle='PDL 开启 · Fusion 开启 · CUDA Graph 开启',tracks=[dict(label='标准推理',**representative(on))]),
 dict(id='pdl',title='PDL 对照 · 四步 DiT',scope='首个 DiT Kernel → 最后一个 DiT Kernel',subtitle='两组均启用 Fusion 和 CUDA Graph；各自以 DiT 首个 Kernel 为 0',tracks=[dict(label='PDL 关闭',**representative(off,True)),dict(label='PDL 开启',**representative(on,True))]),
 dict(id='fusion',title='Fusion 对照 · 完整路径',scope='完整 aux → action',subtitle='两组均关闭 PDL、启用 CUDA Graph；Unfused 拆分 F1–F8',tracks=[dict(label='Unfused',**representative(unfused)),dict(label='Fused',**representative(fused))])])
template=(ROOT/'analysis/timeline.html').read_text(encoding='utf-8')
(ROOT/'kernel_timeline.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')),encoding='utf-8')
(ROOT/'analysis/timeline-summary.json').write_text(json.dumps([{**{k:v for k,v in p.items() if k!='tracks'},'tracks':[{k:v for k,v in t.items() if k!='events'}|{'kernel_count':len(t['events'])} for t in p['tracks']]} for p in data['panels']],ensure_ascii=False,indent=2),encoding='utf-8')
for p in data['panels']:
 for t in p['tracks']:print(p['id'],t['label'],t['request'],len(t['events']),round(t['span'],3),t['lanes'])
