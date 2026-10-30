"""Build auditable summaries from unprofiled measurements and Nsys exports."""
import collections,csv,datetime,json,re,sqlite3,statistics
import torch
from pathlib import Path
R=Path(__file__).resolve().parents[1]; P=R/'reports'; A=R/'analysis';A.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def median(x):return statistics.median(x)
def normalize(s):return re.sub(r'\(bool\)[01]','(bool)X',s)
old=read(R.parent/'flashrt-recipes-controlled-5090-20261030/analysis/execution-data.json')
baseline={};timeline={};validation={}
outputs={};scale_hashes=[]
for label in ['fp8','nvfp4-off','nvfp4-on']:
    walls=[read(P/label/f'wall-{i}/metadata.json') for i in range(3)]
    rounds=[w['timing']['median_ms'] for w in walls]
    baseline[label]=dict(median_ms=median(rounds),rounds_ms=rounds,error=walls[0]['reference_error'])
    tensors=[torch.load(P/label/f'wall-{i}/output.pt',map_location='cpu',weights_only=True).float() for i in range(3)]
    assert all(torch.isfinite(x).all() for x in tensors)
    assert all(torch.equal(tensors[0],x) for x in tensors)
    outputs[label]=tensors[0];scale_hashes.extend(w['frozen_scale_hashes'] for w in walls)
    c=sqlite3.connect(P/label/'trace.sqlite');c.row_factory=sqlite3.Row
    names=dict(c.execute('select id,value from StringIds'))
    ranges=list(c.execute("select start,end,text from NVTX_EVENTS where text like 'REQUEST|%' order by start"));assert len(ranges)==10
    template=next(x for x in old['recipes'] if x['id']==('fp8-fused' if label=='fp8' else 'nvfp4-fused'))
    allrows=[];counts=[];allstage=collections.defaultdict(list);order=[]
    for rg in ranges:
        ks=list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start',(rg['start'],rg['end'])))
        assert len(ks)==len(template['events']),(label,len(ks))
        groups=collections.defaultdict(list)
        # Verify signatures per stage, allowing within-stage PDL launch order variation.
        expected=collections.defaultdict(collections.Counter);actual=collections.defaultdict(collections.Counter)
        for k,t in zip(ks,template['events']):
            stage=t['stage'];stage='Backbone' if stage in ['ViT','VL','DeepStack','LLM'] else stage
            if stage=='Bridge':continue
            expected[stage][(normalize(old['names'][t['n']]),tuple(t['grid']),tuple(t['block']))]+=1
            actual[stage][(normalize(names[k['demangledName']]),tuple(k['grid'+q] for q in 'XYZ'),tuple(k['block'+q] for q in 'XYZ'))]+=1
            groups[stage].append(k)
        for stage in expected:assert expected[stage]==actual[stage],(label,stage,'signature mismatch')
        row=[]
        for stage,items in groups.items():
            span=(max(k['end'] for k in items)-min(k['start'] for k in items))/1e6
            allstage[stage].append(span);row.append(dict(stage=stage,start_ms=(min(k['start'] for k in items)-ks[0]['start'])/1e6,span_ms=span,kernels=len(items)))
        allrows.append(dict(request=rg['text'],span_ms=(max(k['end'] for k in ks)-ks[0]['start'])/1e6,stages=row))
        counts.append(len(ks))
    representative=min(allrows,key=lambda r:abs(r['span_ms']-median(x['span_ms'] for x in allrows)))
    timeline[label]=dict(median_span_ms=median(x['span_ms'] for x in allrows),stage_medians_ms={s:median(v) for s,v in allstage.items()},representative=representative,requests=allrows)
    validation[label]=dict(requests=len(ranges),kernels_per_request=counts,stage_signatures_match=True)
assert all(x==scale_hashes[0] for x in scale_hashes)
assert torch.equal(outputs['nvfp4-off'],outputs['nvfp4-on'])
validation['outputs']=dict(all_finite=True,baseline_rounds_bitwise_equal=True,pdl_output_bitwise_equal=True,frozen_scale_hashes_equal=True)
Q=P/'confirmation' if (P/'confirmation/mechanisms/results.json').exists() else P
full=read(Q/'segments/full.json');local=read(Q/'segments/local.json');cross=read(Q/'segments/cross.json');mechanisms=read(Q/'mechanisms/results.json')
assert len(local)==48
assert full['errors']['native']['max_abs']==0
assert full['errors']['cross-split']['max_abs']==0
assert mechanisms['errors']['native']['max_abs']==0
for stem,prefix,n in [('hybrid-trace','HYBRID',7),('mechanisms-trace','MECHANISM',3)]:
    c=sqlite3.connect(P/(stem+'.sqlite'));c.row_factory=sqlite3.Row
    names=dict(c.execute('select id,value from StringIds'))
    reference=next(x for x in old['recipes'] if x['id']=='nvfp4-fused')
    prefix_length=next(i for i,e in enumerate(reference['events']) if e['stage']=='CrossKV')
    expected=[(normalize(old['names'][e['n']]),tuple(e['grid']),tuple(e['block'])) for e in reference['events'][:prefix_length]]
    rg=list(c.execute('select start,end,text from NVTX_EVENTS where text like ?',(prefix+'|%',)))
    counts=collections.defaultdict(list)
    for r in rg:
        kernels=list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start',(r['start'],r['end'])))
        count=len(kernels)
        actual=[(normalize(names[k['demangledName']]),tuple(k['grid'+q] for q in 'XYZ'),tuple(k['block'+q] for q in 'XYZ')) for k in kernels[:prefix_length]]
        assert expected==actual,(stem,r['text'],'Backbone changed')
        counts[r['text'].split('|')[1]].append(count)
    assert len(counts)==n and all(len(x)==5 and len(set(x))==1 for x in counts.values()),counts
    validation[stem]=dict(requests=len(rg),kernel_counts=dict(counts),backbone_prefix_matches=True)
telemetry={}
for path in P.glob('telemetry*.csv'):
    with path.open() as f:rows=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(f)]
    def number(v):return float(v.split()[0])
    keys=[k for k in rows[0] if k!='timestamp' and k!='pstate']
    selected=rows
    if path.name=='telemetry.csv':
        windows=[(w['start_ns']/1e9,w['end_ns']/1e9) for label in baseline for i in range(3) for w in [read(P/label/f'wall-{i}/metadata.json')]]
        zone=datetime.timezone(datetime.timedelta(hours=8))
        selected=[r for r in rows if any(a<=datetime.datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=zone).timestamp()<=b for a,b in windows)]
    telemetry[path.name]=dict(samples=len(selected),scope='unprofiled baseline measurement windows' if path.name=='telemetry.csv' else 'entire process including setup/trace',metrics={k:dict(min=min(number(r[k]) for r in selected),max=max(number(r[k]) for r in selected),median=median(number(r[k]) for r in selected)) for k in keys})
confirmation={}
if Q!=P:
    with (Q/'telemetry.csv').open() as f:rows=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(f)]
    zone=datetime.timezone(datetime.timedelta(hours=8))
    for label,w in dict(full=full,local=read(Q/'segments/local-window.json'),cross=cross,mechanisms=mechanisms).items():
        selected=[r for r in rows if w['start_ns']/1e9<=datetime.datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=zone).timestamp()<=w['end_ns']/1e9]
        confirmation[label]=dict(samples=len(selected),metrics={k:dict(min=min(number(r[k]) for r in selected),max=max(number(r[k]) for r in selected),median=median(number(r[k]) for r in selected)) for k in keys})
summary=dict(baseline=baseline,gains_ms=dict(total=baseline['fp8']['median_ms']-baseline['nvfp4-on']['median_ms'],without_pdl=baseline['fp8']['median_ms']-baseline['nvfp4-off']['median_ms'],pdl=baseline['nvfp4-off']['median_ms']-baseline['nvfp4-on']['median_ms']),full=full,local=local,cross=cross,mechanisms=mechanisms,timeline=timeline,telemetry=telemetry,confirmation_telemetry=confirmation,source_measurements=Q.relative_to(R).as_posix(),validation=validation)
# Compact samples for the reader; retain complete round-level raw measurements separately.
def compact(x):
    if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ['samples_ms','samples_us']}
    if isinstance(x,list):return [compact(v) for v in x]
    return x
(A/'summary.json').write_text(json.dumps(compact(summary),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(gains=summary['gains_ms'],full={k:v['median_ms'] for k,v in full['timing'].items()},mechanisms={k:v['median_ms'] for k,v in mechanisms['timing'].items()},telemetry=telemetry,validation=validation),indent=2))
