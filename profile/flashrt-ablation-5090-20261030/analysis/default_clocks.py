import csv,json,datetime,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
raw=root/'reports/default-clocks-retest'
offset=raw.joinpath('timezone.txt').read_text().strip()
with (raw/'telemetry.csv').open() as f:
 rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
for r in rows:
 r['unix_ns']=int(datetime.datetime.strptime(r['timestamp']+offset,'%Y/%m/%d %H:%M:%S.%f%z').timestamp()*1e9)
result=[]
for p in sorted(raw.glob('round-*/wall-metadata.json')):
 m=json.loads(p.read_text());rr=[r for r in rows if m['measurement_start_unix_ns']<=r['unix_ns']<=m['measurement_end_unix_ns']]
 assert rr,p
 result.append(dict(round=p.parent.name,timing=m['timings']['frontend'],measurement_start_unix_ns=m['measurement_start_unix_ns'],measurement_end_unix_ns=m['measurement_end_unix_ns'],telemetry=rr))
out=dict(condition='Default clocks restored using nvidia-smi -rgc/-rmc; original 10-call warmup and 50-call benchmark; external telemetry every 50 ms',rounds=result)
(root/'analysis/default-clocks-results.json').write_text(json.dumps(out,indent=2))
for r in result:
 print(r['round'],r['timing']['median_ms'],{k:sorted(set(x[k] for x in r['telemetry'])) for k in r['telemetry'][0] if k not in ['timestamp','unix_ns']})
