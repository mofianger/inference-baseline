"""Isolate DiT-only F1-F8-scope consumers and partial-materialization cost."""
import os,sys,json,argparse,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'profile/flashrt-recipes-controlled-5090-20261030/harness'),str(ROOT/'profile/flashrt-ablation-5090-20261030/harness')]
os.environ['RECIPE']='nvfp4-fused';os.environ['BENCH_PDL']='0'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
from setup import setup,torch,error
from controls import Controls,NAMES
from measure import wall_pairs
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--trace',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 adapter,sample,aux,fn,meta=setup();f=adapter.frontend;ref=fn().clone();K=f._fvk;original=types.SimpleNamespace(**{n:getattr(K,n) for n in dir(K)});ctl=Controls(original)
 states={};errors={}
 for mode in ['native','dit-unfused','attention-materialized']:
  for n in NAMES:setattr(K,n,getattr(original,n))
  targets=[n for n in NAMES if n.startswith('nvfp4_skinny')] if mode=='dit-unfused' else ['nvfp4_skinny_partials_attention'] if mode=='attention-materialized' else []
  for name in targets:setattr(K,name,lambda *args,name=name:ctl.split(name,list(args)))
  f._capture_kernel_dit_graphs(num_inference_timesteps=4,action_horizon=40)
  states[mode]=f._k_dit_graph;y=fn();torch.cuda.synchronize();errors[mode]=error(y,ref);assert torch.isfinite(y).all();print('GRAPH',mode,errors[mode],flush=True)
 assert errors['native']['max_abs']==0
 def run(mode):f._k_dit_graph=states[mode];return fn()
 if a.trace:
  for m in states:run(m)
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
  for m in states:
   for i in range(5):torch.cuda.nvtx.range_push('MECHANISM|'+m+'|'+str(i));run(m);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop();timing={}
 else:timing=wall_pairs({m:lambda m=m:run(m) for m in states})
 (out/'results.json').write_text(json.dumps(dict(metadata=meta,timing=timing,errors=errors,scope='Backbone unchanged; DiT consumers only; PDL off; experimental split implementations'),indent=2))
