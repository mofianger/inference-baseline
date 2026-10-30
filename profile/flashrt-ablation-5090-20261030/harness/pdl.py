import argparse,json,os,time,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
from setup import setup,error,torch

def telemetry():
 return subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate','--format=csv,noheader'],text=True).strip()

with torch.inference_mode():
 adapter,sample,aux,fn,meta=setup()
 t=time.perf_counter()
 while time.perf_counter()-t<5:fn()
 torch.cuda.synchronize();before=telemetry()
 if a.trace:
  torch.cuda.cudart().cudaProfilerStart()
  for i in range(10):
   torch.cuda.nvtx.range_push(f'PDL|{os.environ.get("FLASHRT_GROOT_N17_DIT_PDL")}|{i}')
   y=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()
  values=[]
 else:
  values=[]
  for i in range(150):
   torch.cuda.synchronize();t=time.perf_counter();y=fn();torch.cuda.synchronize();values.append((time.perf_counter()-t)*1000)
 meta.update(pdl=os.environ.get('FLASHRT_GROOT_N17_DIT_PDL','1'),scope='GPU aux/state/noise -> GPU action',warmup_seconds=5,
             wall_ms=values,median_ms=statistics.median(values) if values else None,
             telemetry_before=before,telemetry_after=telemetry(),reference_error=error(y,sample['reference_actions']))
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(meta,indent=2))
 print(json.dumps({k:meta[k] for k in ('pdl','median_ms','telemetry_before','telemetry_after','reference_error')}),flush=True)
