"""Fresh three-view trace with the complete visual embedding contract.

The official BF16 visual model prepares aux. It is measured separately and in
the integrated call; no FlashRT kernel or upstream source is modified.
"""
import argparse, json, os, sys, time, statistics, inspect, linecache
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'deps'))
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--mode', choices=['trace','mapping'], default='trace')
a = p.parse_args()
cfg = json.loads((ROOT/'configs/5090/flashrt_nvfp4.json').read_text())
cfg['options']['calibration_cache'] = str(HERE.parent/'reports/calibration-corrected')
for k,v in cfg['environment'].items(): os.environ[k]=v
sys.path[:0] = [str(ROOT), *cfg['python_paths']]
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from inference_baseline.adapters.flashrt import FlashRTAdapter
from inference_baseline.dataset import load_dataset, move_sample
from inference_baseline.io import revision,sha256

torch.set_num_threads(8); torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
manifest,protocol,cpu=load_dataset(ROOT/'datasets/groot-n17-v3')
samples=[move_sample(s,'cuda:0') for s in cpu]
vc=AutoConfig.from_pretrained(cfg['cosmos'],local_files_only=True).vision_config
vc._attn_implementation='sdpa'
visual=Qwen3VLVisionModel(vc).to(device='cuda',dtype=torch.bfloat16).eval()
index=json.loads((Path(cfg['checkpoint'])/'model.safetensors.index.json').read_text())['weight_map']
prefix='backbone.model.model.visual.'
state={}
for shard in sorted(set(v for k,v in index.items() if k.startswith(prefix))):
 with safe_open(str(Path(cfg['checkpoint'])/shard),framework='pt',device='cpu') as f:
  for k in f.keys():
   if k.startswith(prefix):state[k[len(prefix):]]=f.get_tensor(k)
visual.load_state_dict(state,strict=True); del state

class CompleteAdapter(FlashRTAdapter):
 def aux(self,sample):
  aux=super().aux(sample)
  with torch.profiler.record_function('SUPPORT|Official ViT24 + Merger'):
   image_tokens,_=visual(sample['inputs']['pixel_values'],grid_thw=sample['inputs']['image_grid_thw'])
  mask=aux['visual_pos_masks'].unsqueeze(-1).expand_as(aux['llm_input_embeds'])
  assert int(mask.sum())==image_tokens.numel()
  aux['llm_input_embeds']=aux['llm_input_embeds'].masked_scatter(mask,image_tokens.to(aux['llm_input_embeds'].dtype))
  return aux

def error(x,y):
 x=x.float().flatten();y=y.float().flatten();d=x-y
 return dict(mae=d.abs().mean().item(),rmse=d.square().mean().sqrt().item(),max_abs=d.abs().max().item(),cosine=torch.nn.functional.cosine_similarity(x,y,dim=0).item())

with torch.inference_mode():
 adapter=CompleteAdapter(cfg,protocol,samples)
 sample=next(s for s in samples if s['role']=='evaluation')
 aux=adapter.aux(sample)
 def frontend():return adapter.frontend.infer(sample['inputs']['state'],aux=aux,initial_noise=sample['noise'],num_inference_timesteps=4,action_horizon=40,use_dit_graph=True)
 def integrated():return adapter.infer(sample)
 for _ in range(10):frontend()
 metadata=dict(engine=revision(cfg['engine_source']),torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name(),protocol=protocol.to_dict(),sample=sample['id'],dataset_sha256=sha256(ROOT/'datasets/groot-n17-v3/manifest.json'),runtime=adapter.runtime_stats(),input_shapes={k:list(v.shape) for k,v in sample['inputs'].items()},scope='frontend with complete aux; integrated scope additionally runs official BF16 visual preparation',visual_support='Qwen3VLVisionModel SDPA, all 24 blocks + merger, strict checkpoint load; unmodified FlashRT recalculates DeepStack taps',binaries=adapter.metadata['binaries'])
 timings={}
 for name,fn in [('frontend',frontend),('integrated',integrated)]:
  for _ in range(3):fn()
  ms=[]
  for _ in range(30):
   torch.cuda.synchronize();t=time.perf_counter();y=fn();torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
  timings[name]=dict(samples_ms=ms,median_ms=statistics.median(ms),min_ms=min(ms),max_ms=max(ms))
 metadata['timings']=timings
 metadata['reference_error']=error(y,sample['reference_actions'])
 metadata['acceptance_threshold']=None
 metadata['repeat_error']=error(frontend(),frontend())
 checks=[]
 for s in samples:
  z=adapter.infer(s);torch.cuda.synchronize();checks.append(dict(sample=s['id'],role=s['role'],**error(z,s['reference_actions'])))
 metadata['dataset_errors']=checks
 # Restore the evaluation observation before tracing or mapping.
 frontend();torch.cuda.synchronize()
 if a.mode=='trace':
  for name,fn in [('frontend',frontend),('integrated',integrated)]:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for i in range(10):
     with torch.profiler.record_function(f'REQUEST|{name}|{i}'):
      y=fn();torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/f'{name}.json'))
 else:
  catalog={};active=[None]
  def hook(frame,event,arg):
   path=frame.f_code.co_filename
   if '/flash_rt/' not in path:return None
   if event=='line':
    if active[0]:active[0].__exit__(None,None,None)
    labels=[];f=frame;scalars={}
    while f:
     if '/flash_rt/' in f.f_code.co_filename:
      loc=f.f_locals
      labels.append(f.f_code.co_name+':'+','.join(str(k)+'='+str(loc[k]) for k in ('li','j','step','name','what') if k in loc and isinstance(loc[k],(int,str))))
      for key in ('S','Sv','Se','Sa','D','FF','N','M','K','Skv_text','Skv_image','Nout','Sper','T','n','k','splits','kv','kv_ld'):
       if key in loc and isinstance(loc[key],int):scalars.setdefault(key,loc[key])
     f=f.f_back
    rel=path[path.index('flash_rt/'):];line=frame.f_lineno
    label='MAP|'+rel+':'+str(line)+'|'+'|'.join(labels)
    catalog[label]=dict(file=rel,line=line,code=''.join(linecache.getline(path,i) for i in range(line,min(line+5,len(linecache.getlines(path))+1))).strip(),scalars=scalars)
    active[0]=torch.profiler.record_function(label);active[0].__enter__()
   return hook
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   sys.settrace(hook)
   f=adapter.frontend;f._kbb_forward(0);f._cross_kv_fwd(0);st,step=f._kdit_fwd;st(0)
   for i in range(4):step(i,0)
   sys.settrace(None)
   if active[0]:active[0].__exit__(None,None,None)
   torch.cuda.synchronize()
  prof.export_chrome_trace(str(out/'mapping.json'))
  (out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 print(json.dumps({k:v for k,v in metadata.items() if k not in ('binaries','timings')},indent=2),flush=True)
 print('TIMINGS', {k:v['median_ms'] for k,v in timings.items()},flush=True)
