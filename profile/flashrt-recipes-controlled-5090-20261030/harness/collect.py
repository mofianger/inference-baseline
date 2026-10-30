import os,argparse,time,json,statistics,hashlib
from pathlib import Path
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['wall','trace','map'],default='wall');p.add_argument('--out',required=True);a=p.parse_args()
from setup import setup,torch,error,cfg
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 adapter,sample,aux,fn,meta=setup();f=adapter.frontend
 meta.update(recipe=os.environ['RECIPE'],scope='GPU complete aux/state/noise -> GPU action',adapter=adapter.metadata)
 t=time.perf_counter()
 while time.perf_counter()-t<5:fn()
 if a.mode=='wall':
  ms=[];meta['start_ns']=time.time_ns()
  for _ in range(150):
   torch.cuda.synchronize();t=time.perf_counter();y=fn();torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
  meta['end_ns']=time.time_ns();meta['timing']=dict(median_ms=statistics.median(ms),samples_ms=ms)
 elif a.mode=='trace':
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
  for i in range(10):
   torch.cuda.nvtx.range_push('REQUEST|'+os.environ['RECIPE']+'|'+str(i));y=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()
 else:
  from mapping import capture
  capture(f,sample,aux,out);y=fn()
 meta['reference_error']=error(y,sample['reference_actions']);torch.save(y.cpu(),out/'output.pt')
 scales=Path(__file__).resolve().parents[1]/'reports/shared-scales'
 meta['frozen_scale_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scales.glob('*.json')}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));print('DONE',os.environ['RECIPE'],meta.get('timing',{}).get('median_ms'),flush=True)
