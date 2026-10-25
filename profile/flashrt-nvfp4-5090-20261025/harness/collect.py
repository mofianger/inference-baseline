"""Profile the existing FlashRT frontend; adapter embeddings are prepared outside ranges."""
import argparse,json,os,time,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',default='torch');a=p.parse_args()
root=Path(__file__).resolve().parents[3]
config=json.loads((root/'configs/5090/flashrt_nvfp4.json').read_text())
for k,v in config['environment'].items():os.environ[k]=v
import sys
sys.path[:0]=[str(root),*config['python_paths']]
import torch
from inference_baseline.dataset import load_dataset,move_sample
from inference_baseline.adapters import create
from inference_baseline.io import revision,sha256
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8);torch.set_num_interop_threads(1)
manifest,protocol,cpu=load_dataset(root/'datasets/groot-n17-v3')
with torch.inference_mode():
 samples=[move_sample(s,torch.device('cuda:0')) for s in cpu]
 adapter=create('flashrt',config,protocol,samples)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sample=next(s for s in samples if s['role']=='evaluation')
 aux=adapter.aux(sample);torch.cuda.synchronize()
 def invoke():return adapter.frontend.infer(sample['inputs']['state'],aux=aux,initial_noise=sample['noise'],num_inference_timesteps=4,action_horizon=40,use_dit_graph=True)
 for _ in range(10):invoke()
 torch.cuda.synchronize();reference=invoke().clone();torch.cuda.synchronize()
 ms=[]
 for _ in range(30):
  torch.cuda.synchronize();t=time.perf_counter();y=invoke();torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
 meta={'scope':'FlashRT frontend, prepared embeddings excluded; internal copies, sync, output cast included','sample':sample['id'],'baseline_ms':ms,'median_ms':statistics.median(ms),'engine':revision(config['engine_source']),'runtime':adapter.runtime_stats(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'sm_count':torch.cuda.get_device_properties(0).multi_processor_count,'dataset_sha256':sha256(root/'datasets/groot-n17-v3/manifest.json')}
 if a.mode=='torch':
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   for i in range(10):
    with torch.profiler.record_function(f'FLASHRT_REQUEST_{i}'):
     y=invoke();torch.cuda.synchronize()
  prof.export_chrome_trace(str(out/'trace.json'))
 elif a.mode=='nsys':
  torch.cuda.cudart().cudaProfilerStart()
  for i in range(3):
   torch.cuda.nvtx.range_push(f'FLASHRT_REQUEST_{i}');y=invoke();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()
 meta['repeat_max_abs']=float((y-reference).abs().max());(out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
