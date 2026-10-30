"""Auditable results tables from unprofiled measurements and independent Nsys checks."""
import json,sqlite3,statistics,hashlib,html,csv
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];R=RUN/'reports'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
pd=[read(p) for p in sorted((R/'pdl').glob('run-*.json'))]
fusion=read(R/'fusion-calibrated/full.json');local=read(R/'fusion-calibrated/local.json');prec=read(R/'precision-calibrated/precision.json')

def compact(x):
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ('samples_us','samples_ms','wall_ms','binaries')}
 if isinstance(x,list):return [compact(v) for v in x]
 return x

def nsys(path,prefix):
 c=sqlite3.connect(path);c.row_factory=sqlite3.Row
 ranges=c.execute('select start,end,text from NVTX_EVENTS where text like ? order by start',(prefix+'%',)).fetchall()
 result=[]
 for rr in ranges:
  ks=[dict(x) for x in c.execute('select k.start,k.end,k.streamId,k.registersPerThread,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.start<? order by k.start',(rr['start'],rr['end']))]
  assert ks,(path,rr['text'])
  result.append(dict(range=rr['text'],kernels=len(ks),span_us=(max(k['end'] for k in ks)-ks[0]['start'])/1000,entries=ks))
 return result

trace={}
for key,file,prefix in [('pdl_off','pdl/trace-0.sqlite','PDL|'),('pdl_on','pdl/trace-1.sqlite','PDL|'),('local','local-trace.sqlite','LOCAL|'),('fusion','fusion-trace.sqlite','FUSION|'),('precision','precision-trace.sqlite','PRECISION|')]:
 trace[key]=nsys(R/file,prefix)
expected=read(RUN.parent/'flashrt-nvfp4-5090-20261030/analysis/kernels.json');expected=[x for x in expected if 'memcpy32_post' not in x['name']]
for key in ['pdl_off','pdl_on']:
 for rr in trace[key]:
  assert len(rr['entries'])==len(expected)
  for k,e in zip(rr['entries'],expected):
   assert [k['gridX'],k['gridY'],k['gridZ']]==e['grid']
   assert [k['blockX'],k['blockY'],k['blockZ']]==e['block']
  dit=[k for k,e in zip(rr['entries'],expected) if e['stage'].startswith('Step')]
  rr['four_steps_us']=(max(k['end'] for k in dit)-min(k['start'] for k in dit))/1000
for rr in trace['local']:
 group,mode=rr['range'].split('|')[1:3]
 assert rr['kernels']>=1
 if mode=='fused':assert rr['kernels']==1,(group,rr['kernels'])

pdl={mode:dict(median_ms=statistics.median(v['median_ms'] for v in pd if v['pdl']==mode),round_median_ms=[v['median_ms'] for v in pd if v['pdl']==mode],errors=[v['reference_error'] for v in pd if v['pdl']==mode]) for mode in ['0','1']}
def gain(off,on):return dict(saved=off-on,reduction_pct=(off-on)/off*100,speedup=off/on)
pdl['gain']=gain(pdl['0']['median_ms'],pdl['1']['median_ms'])
fusion_gain=gain(fusion['timing']['unfused']['median_ms'],fusion['timing']['fused']['median_ms'])
telemetry={}
for p in R.glob('telemetry-*.csv'):
 with p.open() as f:
  rr=list(csv.DictReader(f))
 telemetry[p.name]=dict(samples=len(rr),sm_clocks=sorted(set(r.get(' clocks.current.sm [MHz]','') for r in rr)),memory_clocks=sorted(set(r.get(' clocks.current.memory [MHz]','') for r in rr)))
report=dict(pdl=pdl,fusion=compact(fusion),fusion_gain=fusion_gain,local=compact(local),precision=compact(prec),telemetry=telemetry,nsys={key:[{k:v for k,v in x.items() if k!='entries'} for x in rr] for key,rr in trace.items()},raw_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.suffix in ('.sqlite','.json','.csv') and 'build' not in p.parts and 'source' not in p.parts})
(RUN/'analysis/results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
# Keep native names per task for checking actual dispatch/fusion in Nsys.
dispatch={key:{r['range']:[{k:v for k,v in e.items() if k not in ('start','end')} for e in r['entries']] for r in rr if r['range'].endswith('|0')} for key,rr in trace.items()}
(RUN/'analysis/kernel-dispatch.json').write_text(json.dumps(dispatch,ensure_ascii=False,indent=2),encoding='utf-8')

def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
fmt=lambda x:f'{x:.3f}'
body='''<header><small>INFERENCE BASELINE · CONTROLLED ABLATIONS</small><h1>FlashRT：优化到底省了多少时间？</h1><p>RTX 5090 · GR00T N1.7 · 3 views · batch 1 · 4 步去噪 · 40 个 action</p></header>'''
body+='<div class="notice">输入为已在 GPU 就绪的 aux、state 和固定噪声。权重、图像 embedding 准备不计时。实际工作频率与先前默认频率的 Profile 不同，请在本页对照组之间比较。</div>'
body+='<section><h2>01 · PDL 开关</h2>'+table(['PDL','整条路径中位数 ms','三轮中位数 ms'],[[m,fmt(pdl[m]['median_ms']),', '.join(fmt(x) for x in pdl[m]['round_median_ms'])] for m in ['0','1']])+f"<p>开启后节省 <b>{pdl['gain']['saved']:.3f} ms</b>，延迟下降 <b>{pdl['gain']['reduction_pct']:.2f}%</b>。Graph、Fusion 和精度保持一致。</p></section>"
body+='<section><h2>02 · Fusion：整条路径</h2>'+table(['配置','中位数 ms','六轮中位数 ms'],[[m,fmt(v['median_ms']),', '.join(fmt(x) for x in v['round_median_ms'])] for m,v in fusion['timing'].items()])+f"<p>融合节省 <b>{fusion_gain['saved']:.3f} ms</b>，延迟下降 <b>{fusion_gain['reduction_pct']:.2f}%</b>。两边均关闭 PDL、开启 Graph；Unfused 仅拆分 F1–F8，Attention 主体等其他融合保留。</p><p>Unfused 与 Fused action 的 MAE：{fusion['error_vs_fused']['unfused']['mae']:.6f}；最大误差：{fusion['error_vs_fused']['unfused']['max_abs']:.6f}。仅报告误差，未设置通过阈值。</p></section>"
body+='<section><h2>03 · Fusion：局部计算段</h2><p>单位 μs。包含完成同一计算任务的全部 Kernel；缓存扰动在计时外。负收益也保留。</p>'
rows=[]
for r in local:
 for c,tt in r['timing'].items():
  f,u=tt['fused']['median_us'],tt['unfused']['median_us'];rows.append([r['group'],'热缓存' if c=='warm' else '128 MiB 扰动',fmt(f),fmt(u),fmt(u-f),f'{u/f:.2f}×'])
body+=table(['融合组','缓存条件','Fused','Unfused','节省','加速'],rows)+'</section>'
body+='<section><h2>04 · 精度：相同计算任务</h2><p>输入输出边界一致，允许不同高性能实现。包含激活量化、必要归约和格式转换；权重转换在计时外。FP8 小行数按接口要求补齐，计算成本包含在内。</p>'
for r in prec['results']:
 body+='<h3>'+html.escape(r['task'])+'</h3><p class="note">'+html.escape(r.get('boundary',''))+'</p>'
 rows=[]
 for key in r['timing']['warm']:
  rows.append([key,fmt(r['timing']['warm'][key]['median_us']),fmt(r['timing']['l2_pressure_128MiB'][key]['median_us']),f"{r['errors'][key]['mae']:.6f}",f"{r['errors'][key]['cosine']:.6f}"])
 body+=table(['实现','热缓存 μs','缓存扰动 μs','MAE','余弦相似度'],rows)
body+='</section><section><h2>05 · 如何测量</h2><p>热缓存结果采用 64 次任务批量摊销，减去匹配的输入恢复对照；预热后交替测 6 轮，每轮 10 个批次。它是任务耗时的估计值，原始批量成本和恢复成本均保留。缓存扰动结果采用逐任务 Event 窗口，40 次 × 6 轮，包含事件边界开销；两列使用不同估计方法，不能直接相减解释缓存收益。整条路径采用同步 wall latency。Nsys 独立运行，只用于验证执行路径和解释差异。</p><p>128 MiB 扰动属于缓存压力对照，不等同于实际推理中的缓存状态。局部差值不相加作为整条路径收益。读数包含事件边界开销，亚微秒差距应结合轮间波动判断。</p><p><a href="REPORT.md">完整报告与复现说明</a> · <a href="analysis/results.json">结构化结果</a> · <a href="analysis/kernel-dispatch.json">Nsys Kernel 调用</a></p></section>'
style='''body{margin:0;background:#f4f5ef;color:#253d37;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:32px}header{border-top:4px solid #317859;padding:22px 0}small{letter-spacing:2px;color:#317859}h1{font-size:36px;line-height:1.3}h2{font-size:23px}h3{margin-top:25px}section{background:#fffefa;padding:25px;margin:24px 0;border:1px solid #dce1d4}.notice{background:#e8eee0;padding:18px;border-left:3px solid #317859}.note{font-size:12px;color:#63746c}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e1e5d9;font-variant-numeric:tabular-nums}th{background:#edf1e5;white-space:nowrap}a{color:#317859}b{color:#317859}@media(max-width:650px){main{padding:14px}section{padding:15px}h1{font-size:27px}}'''
(RUN/'ablation_results.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlashRT 优化收益消融</title><style>'+style+'</style><main>'+body+'</main></html>',encoding='utf-8')
print(json.dumps(dict(pdl=pdl['gain'],fusion=fusion_gain,local_cases=len(local),precision_cases=len(prec['results']),nsys_ranges={k:len(v) for k,v in trace.items()}),indent=2))
