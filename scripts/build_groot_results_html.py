"""Build a standalone, offline reader-facing report from recorded measurements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'reports/2026-10-25-5090-groot-new-recipes'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
NOTES = read(ROOT/'scripts/groot_recipe_notes.json')

def measured(key, path, history=False):
    d = read(path)
    assert d['dataset_manifest_sha256'] == '562a86fc238ca10adcc2b623be09bf796b74f1403fbdd5ab6f6d1785335c2297'
    n = NOTES[key]
    unique = {x['sample_id']: x for x in d['correctness']}
    return dict(n, id=key,
        mean=d['latency']['mean_ms'], p95=d['latency']['p95_ms'],
        error=max(x['max_abs'] for x in unique.values()), mae=sum(x['mae'] for x in unique.values()) / len(unique),
        source=path.relative_to(ROOT).as_posix(), commit=d['engine_source']['commit'],
        date=d['timestamp_utc'][:10], history=history, diagnostic=key.endswith('-eager'), status='measured')

rows = [measured(x['recipe'], NEW/x['source']) for x in read(NEW/'run-index.json') if x['status']=='measured']
for key, filename in [('flashrt-nvfp4','flashrt_nvfp4'),('flashrt-fp8','flashrt_fp8'),('embodiinfer-bf16','embodiinfer_optimized')]:
    rows.append(measured(key, ROOT/f'reports/2026-10-23-5090-optimized/{filename}.json'))
for key, filename, history in [('phyai-bf16','phyai',False),('reference','reference',False)]:
    rows.append(measured(key, ROOT/f'reports/2026-10-23-5090-smoke/{filename}.json',history))
assert len(rows)==len(NOTES)==17
payload = json.dumps(rows,ensure_ascii=False).replace('</','<\/')
template=(ROOT/'scripts/templates/groot_results.html').read_text(encoding='utf-8')
for token, filename in [('__APPENDIX__','groot_model_appendix.html'),('__MODEL_JS__','groot_model_explorer.js'),('__CONTROLS__','groot_result_controls.html'),('__FILTER_JS__','groot_result_filters.js'),('__EXTRA_CSS__','groot_results_extra.css')]:
    template = template.replace(token, (ROOT/'scripts/templates'/filename).read_text(encoding='utf-8-sig'))
target=ROOT/'groot_n17_5090_results.html'
target.write_text(template.replace('__DATA__',payload),encoding='utf-8')
print(f'Built {target.name}: {len(rows)} measured recipes')
