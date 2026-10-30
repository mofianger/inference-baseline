import os,argparse,json,types
from pathlib import Path
os.environ['FLASHRT_GROOT_N17_DIT_PDL']='0'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
from setup import setup,error,torch
from controls import Controls,NAMES
from measure import graph,paired_graphs,wall_pairs
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 adapter,sample,aux,fn,meta=setup();f=adapter.frontend
 original=types.SimpleNamespace(**{n:getattr(f._fvk,n) for n in dir(f._fvk)})
 ctl=Controls(original)
 for name in NAMES:setattr(f._fvk,name,getattr(ctl,name))
 expected=fn().clone()
 # An eager discovery pass observes original operations without timing it.
 f._kbb_vit_h.copy_(aux['pixel_features'].half().reshape(f._S_vit,1024))
 f._kbb_llm_h.copy_(aux['llm_input_embeds'].half().reshape(f.Se,2048))
 ctl.observe=True;f._kbb_forward(0)
 f._ck_bb_src.copy_(f._kbb_vlsa_h.reshape(f.Se,2048).half());f._k_state_in.copy_(sample['inputs']['state'].reshape(1,132).bfloat16());f._k_actions.copy_(sample['noise'].reshape(40,132).bfloat16())
 f._cross_kv_fwd(0);st,step=f._kdit_fwd;st(0)
 for i in range(4):step(i,0)
 ctl.observe=False;torch.cuda.synchronize()
 torch.save({k:dict(name=v['name'],args=v['args'],buffers={i:t.cpu() for i,t in v['buffers'].items()},outputs=v['outputs'],mutations=v['mutations'],rows=v['rows'],n=v['n']) for k,v in ctl.records.items()},out/'fixtures.pt')
 results=[]
 for key,rec in ctl.records.items():
  args=rec['args'];name=rec['name'];copies={i:rec['buffers'][i].clone() for i in rec['mutations']}
  def reset():
   for i,x in copies.items():rec['buffers'][i].copy_(x)
  def fused():getattr(original,name)(*args[:-1],torch.cuda.current_stream().cuda_stream)
  def unfused():ctl.split(name,[*args[:-1],torch.cuda.current_stream().cuda_stream])
  reset();fused();ref={i:rec['buffers'][i].clone() for i in rec['outputs']}
  reset();unfused();torch.cuda.synchronize()
  err={str(i):dict(**error(rec['buffers'][i],v),exact_fraction=(rec['buffers'][i]==v).float().mean().item(),dtype=str(v.dtype)) for i,v in ref.items()}
  assert all(torch.isfinite(rec['buffers'][i].float()).all() for i in ref)
  timing=paired_graphs({'fused':fused,'unfused':unfused},reset)
  results.append(dict(group=key,operation=name,rows=rec['rows'],n=rec['n'],error=err,timing=timing))
  print('LOCAL',key,json.dumps({c:{k:v['median_us'] for k,v in r.items()} for c,r in timing.items()}),flush=True)
 (out/'local.json').write_text(json.dumps(results,indent=2))
 # Recapture both existing compute closures, preserving frontend copies and return.
 states={}
 for mode in ['fused','unfused']:
  ctl.mode=mode
  f._capture_backbone_graph()
  f._capture_kernel_dit_graphs(num_inference_timesteps=4,action_horizon=40)
  states[mode]=(f._kbb_graph,f._k_dit_graph)
 def run(mode):
  f._kbb_graph,f._k_dit_graph=states[mode]
  return fn()
 fullerr={m:error(run(m),expected) for m in states}
 baselineerr={m:error(run(m),sample['reference_actions']) for m in states}
 if a.trace:
  for m in states:run(m)
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
  for m in states:
   for i in range(5):
    torch.cuda.nvtx.range_push(f'FUSION|{m}|{i}');run(m);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop();timings={}
 else:timings=wall_pairs({m:lambda m=m:run(m) for m in states})
 meta.update(scope='aux to action',pdl=False,fusion_scope='F1-F8; retains library attention and other fusions',timing=timings,error_vs_fused=fullerr,reference_error=baselineerr)
 (out/'full.json').write_text(json.dumps(meta,indent=2));print('FULL',json.dumps({m:v['median_ms'] for m,v in timings.items()}),fullerr,flush=True)
