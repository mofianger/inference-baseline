import os,sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'profile/flashrt-recipes-controlled-5090-20261030/harness'),str(ROOT/'profile/flashrt-ablation-5090-20261030/harness')]
# Select the frozen-calibration setup, before importing the older helper paths.
sys.path.remove(str(ROOT/'profile/flashrt-recipes-controlled-5090-20261030/harness'));sys.path.insert(0,str(ROOT/'profile/flashrt-recipes-controlled-5090-20261030/harness'))
os.environ['RECIPE']='nvfp4-fused';os.environ['BENCH_PDL']='0'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
from setup import setup,torch,error
from controls import Controls
from measure import graph,paired_graphs,wall_pairs
from hybrid import Hybrid
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--trace',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 adapter,sample,aux,fn,meta=setup();f=adapter.frontend;ref=fn().clone();ctl=Controls(f._fvk);h=Hybrid(f,ctl,out)
 f._ck_bb_src.copy_(f._kbb_vlsa_h.reshape(f.Se,2048).half());f._k_state_in.copy_(sample['inputs']['state'].reshape(1,132).bfloat16());f._k_actions.copy_(sample['noise'].reshape(40,132).bfloat16())
 h.observe=True;h.all('native',0);h.observe=False
 if args.smoke:
  for part in ['attention','o','ffn']:
   rec=h.snapshots[(0,0,part)]
   for precision in ['FP8','NVFP4']:
    f._infer_bufs['dit_h'].copy_(rec['h'])
    if part=='o':ctl.view(f._dit_attn.get_slot_ptrs('dit_cross',0)['O'],[41,1536],torch.bfloat16).copy_(rec['o'])
    if part=='ffn':f._infer_bufs['dit_xn'].copy_(rec['xn'])
    y=h.local(part,0,0,precision);torch.cuda.synchronize();assert torch.isfinite(y).all();print('SMOKE',part,precision,flush=True)
  raise SystemExit(0)
 modes={'native':set(),'cross-fp8':set(),'cross-split':set(),'attention-fp8':{'attention'},'o-fp8':{'o'},'ffn-fp8':{'ffn'},'all-segments-fp8':{'attention','o','ffn'}}
 graphs={};errors={}
 for mode,active in modes.items():
  h.active=active;h.set_kv(mode in ['cross-fp8','cross-split'])
  graphs[mode]=graph(lambda mode=mode:h.all(mode,torch.cuda.current_stream().cuda_stream))
  f._k_dit_graph=graphs[mode];y=fn();torch.cuda.synchronize();assert torch.isfinite(y).all();errors[mode]=error(y,ref)
  print('GRAPH',mode,errors[mode],flush=True)
 assert errors['native']['max_abs']==0,errors['native']
 def run(mode):f._k_dit_graph=graphs[mode];return fn()
 measure_start=time.time_ns()
 if args.trace:
  for mode in modes:run(mode)
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
  for mode in modes:
   for i in range(5):torch.cuda.nvtx.range_push('HYBRID|'+mode+'|'+str(i));run(mode);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop();timing={}
 elif (out/'full.json').exists():timing=json.loads((out/'full.json').read_text())['timing']
 else:timing=wall_pairs({mode:lambda mode=mode:run(mode) for mode in modes})
 (out/'full.json').write_text(json.dumps(dict(metadata=meta,timing=timing,errors=errors,start_ns=measure_start,end_ns=time.time_ns(),scope='same aux->action boundary; replacement retains bridge norms/quantization; all PDL off'),indent=2))
 if args.trace:raise SystemExit(0)
 h.active=set();h.set_kv(False);h.cross('native',0)
 local=[];local_start=time.time_ns()
 for step in ([] if (out/'local.json').exists() else [0,1,2,3]):
  for li in [0,1,2,31]:
   for part in ['attention','o','ffn']:
    rec=h.snapshots[(step,li,part)]
    def reset():
     f._infer_bufs['dit_h'].copy_(rec['h'])
     if part=='o':
      slots=f._dit_attn.get_slot_ptrs('dit_self' if li%2 else 'dit_cross',(li-1)//2 if li%2 else li//2);ctl.view(slots['O'],[41,1536],torch.bfloat16).copy_(rec['o'])
     if part=='ffn':f._infer_bufs['dit_xn'].copy_(rec['xn'])
    funcs={p:lambda p=p:h.local(part,li,step,p) for p in ['FP8','NVFP4']}
    reset();reference=funcs['FP8']().clone();reset();y=funcs['NVFP4']();torch.cuda.synchronize();err=error(y,reference);assert torch.isfinite(y).all()
    result=paired_graphs(funcs,reset);local.append(dict(step=step,layer=li,part=part,error=err,timing=result))
    print('LOCAL',step,li,part,{k:v['median_us'] for k,v in result['warm'].items()},flush=True)
 if local:
  (out/'local.json').write_text(json.dumps(local,indent=2))
  (out/'local-window.json').write_text(json.dumps(dict(start_ns=local_start,end_ns=time.time_ns())))
 # Native KV collection boundary, each implementation can retain its own layout.
 kvgraphs={};kvoutputs={}
 for mode in ['native','cross-fp8','cross-split']:
  h.set_kv(mode!='native');h.cross(mode,0);torch.cuda.synchronize();kvoutputs[mode]=torch.cat([t.flatten() for pair in zip(f._dit_cross_K,f._dit_cross_V) for t in pair]).clone()
  kvgraphs[mode]=graph(lambda mode=mode:h.cross(mode,torch.cuda.current_stream().cuda_stream))
 def cross_call(mode):
  h.set_kv(mode!='native');h.cross(mode,torch.cuda.current_stream().cuda_stream)
 cross_start=time.time_ns();kvtiming=paired_graphs({m:lambda m=m:cross_call(m) for m in kvgraphs});kverrors={m:error(t,kvoutputs['native']) for m,t in kvoutputs.items()}
 (out/'cross.json').write_text(json.dumps(dict(timing=kvtiming,errors=kverrors,start_ns=cross_start,end_ns=time.time_ns(),scope='backbone features -> all KV; physical output layout native to variant'),indent=2))
