"""Build an aux-only execution atlas from Nsight Systems SQLite timestamps.

Reuse audited logical I/O descriptions only after exact symbol/order/grid/block
validation. No PyTorch timing is substituted into the Nsys timeline.
"""
import base64
import gzip
import hashlib
import json
import sqlite3
import statistics as st
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
OLD = RUN.parent/'flashrt-nvfp4-5090-20261030'
R = RUN/'reports'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
old = read(OLD/'analysis/kernels.json')
symbols = read(R/'symbols.json')
meta = read(R/'wall-metadata.json')
capture_meta = read(R/'nsys-metadata.json')
assert meta['engine'] == capture_meta['engine']
assert meta['dataset_sha256'] == capture_meta['dataset_sha256']


def load(mode):
    db = sqlite3.connect(R/f'frontend-{mode}.sqlite')
    db.row_factory = sqlite3.Row
    ranges = db.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'REQUEST|frontend|%' ORDER BY start").fetchall()
    assert len(ranges) == 10
    runs, boundary, counts = [], [], []
    for rr in ranges:
        kernels = [dict(x) for x in db.execute('SELECT k.*,s.value AS mangled FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.mangledName WHERE start>=? AND start<? ORDER BY start', (rr['start'], rr['end']))]
        copies = [dict(x) for x in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND start<? ORDER BY start', (rr['start'], rr['end']))]
        graphcopies = [x for x in copies if x['graphNodeId'] is not None]
        assert len(kernels) == 1578 and len(graphcopies) == 8 and len(copies) == 13
        for k in kernels:
            k['name'] = symbols[k['mangled']]
            k['type'] = 'kernel'
        for k in graphcopies:
            k['type'] = 'memcpy'
            k['name'] = f"D2D graph memcpy ({k['bytes']} bytes)"
        merged = sorted(kernels+graphcopies, key=lambda k:k['start'])
        assert len(merged) == len(old)
        for k, o in zip(merged, old):
            if k['type'] == 'kernel':
                assert k['name'] == o['name'], (mode, o['i'], k['name'], o['name'])
                assert [k[f'grid{a}'] for a in 'XYZ'] == o['grid']
                assert [k[f'block{a}'] for a in 'XYZ'] == o['block']
            else:
                assert 'memcpy32_post' in o['name']
                assert k['bytes'] == (3072 if o['grid'][0] == 3 else 122880)
        runs.append(merged)
        boundary.append([x for x in copies if x['graphNodeId'] is None])
        counts.append(dict(kernels=len(kernels), graph_copies=len(graphcopies), boundary_copies=len(copies)-len(graphcopies)))
    spans = [(max(k['end'] for k in rr)-rr[0]['start'])/1000 for rr in runs]
    pick = min(range(10), key=lambda i:abs(spans[i]-st.median(spans)))
    diagnostics = [dict(x) for x in db.execute('SELECT severity,text FROM DIAGNOSTIC_EVENT')]
    return dict(runs=runs, boundary=boundary, counts=counts, spans=spans, pick=pick, diagnostics=diagnostics)


hw, sw = load('hw'), load('sw')
runs, pick = hw['runs'], hw['pick']
ks = runs[pick]
nodes = []
for i, (k, o) in enumerate(zip(ks, old)):
    n = dict(o)
    ds = [(rr[i]['end']-rr[i]['start'])/1000 for rr in runs]
    n.update(name=k['name'], s=(k['start']-ks[0]['start'])/1000,
             d=(k['end']-k['start'])/1000, median=st.median(ds), min=min(ds), max=max(ds),
             stream=k['streamId'], graph_node_id=k['graphNodeId'], event_type=k['type'])
    if k['type'] == 'memcpy':
        n.update(grid=[], block=[], note='Nsys 将本项记录为 Graph 内 D2D Memcpy；保留旧节点编号。它没有 Kernel Grid/Block 属性。', fusion='在 GPU 内复制状态或动作 token', precision='BF16 copy')
    nodes.append(n)

groups = []
for n in nodes:
    if not groups or groups[-1]['label'] != n['group']:
        groups.append(dict(label=n['group'], stage=n['stage'], ids=[]))
    n['g'] = len(groups)-1
    groups[-1]['ids'].append(n['i'])
for g in groups:
    ids = g['ids']
    g['span'] = st.median((max(rr[i]['end'] for i in ids)-min(rr[i]['start'] for i in ids))/1000 for rr in runs)
    g['sum'] = st.median(sum(rr[i]['end']-rr[i]['start'] for i in ids)/1000 for rr in runs)


def union(intervals):
    end, total = -float('inf'), 0
    for a, b in sorted(intervals):
        total += max(0, b-max(a, end))
        end = max(end, b)
    return total


stages = []
for stage in dict.fromkeys(n['stage'] for n in nodes):
    ids = [n['i'] for n in nodes if n['stage'] == stage]
    chunks = [g['ids'] for g in groups if g['stage'] == stage]
    stages.append(dict(stage=stage, ids=ids, span=None if stage=='Bridge' else st.median(union([(min(rr[i]['start'] for i in cc), max(rr[i]['end'] for i in cc)) for cc in chunks])/1000 for rr in runs), sum=st.median(sum(rr[i]['end']-rr[i]['start'] for i in ids)/1000 for rr in runs)))


def compare(trace):
    overlaps = [(rr[211]['end']-rr[212]['start'])/1000 for rr in trace['runs']]
    return dict(gpu_spans_us=trace['spans'], median_gpu_span_us=st.median(trace['spans']),
                gate_up_signed_overlap_us=overlaps, gate_up_overlap_count=sum(x>0 for x in overlaps))


boundary = [dict(s=(k['start']-ks[0]['start'])/1000, d=(k['end']-k['start'])/1000,
                 bytes=k['bytes'], stream=k['streamId']) for k in hw['boundary'][pick]]
meta['dataset_errors'] = [dict(sample=meta['sample'], **meta['reference_error'])]
meta['profiler'] = dict(name='Nsight Systems', version='2025.3.2.474', trace='cuda-hw,nvtx', graph_trace='node', capture_range='cudaProfilerApi', driver='580.95.05', control_trace='cuda,nvtx')
summary = dict(metadata=meta, kernel_count=1578, graph_memcpy_count=8, boundary_memcpy_count=5,
               mapped_compute=1575, node_count=len(nodes), group_count=len(groups), selected_replay=pick+1,
               gpu_span_us=st.median(hw['spans']), gpu_spans_us=hw['spans'], stage_stats=stages,
               comparison=dict(cuda_hw=compare(hw), cuda_sw=compare(sw)),
               validation='10 runs in each Nsys capture: exact canonical kernel names/order/grid/block match prior audited source map. Eight graph copies matched by sequence and byte size. All 10 request ranges complete.',
               diagnostics=hw['diagnostics'], counts=hw['counts'],
               hashes={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['frontend-hw.sqlite','frontend-hw.nsys-rep','frontend-sw.sqlite','frontend-sw.nsys-rep','wall-metadata.json','nsys-metadata.json','symbols.json']})
data = dict(summary=summary, nodes=nodes, groups=groups, integrated=[], boundary_copies=boundary)
for file, obj in [('summary.json',summary), ('kernels.json',nodes)]:
    (RUN/'analysis'/file).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
packed = base64.b64encode(gzip.compress(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode())).decode()
template = (RUN/'harness/site.html').read_text(encoding='utf-8')
(RUN/'flashrt_nvfp4_nsys.html').write_text(template.replace('__DATA__',packed),encoding='utf-8')
print(json.dumps(dict(wall_ms=meta['timings']['frontend']['median_ms'], gpu_ms=summary['gpu_span_us']/1000, selected=pick+1, comparison=summary['comparison']),indent=2))
