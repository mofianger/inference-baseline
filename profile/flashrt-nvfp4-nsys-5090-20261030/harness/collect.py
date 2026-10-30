"""Nsight Systems collection: GPU-ready aux to action, three views.

The official BF16 visual model prepares aux. It is measured separately and in
the integrated call; no FlashRT kernel or upstream source is modified.
"""
import argparse, json, os, sys, time, statistics, inspect, linecache
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'profile/flashrt-nvfp4-5090-20261030/harness/deps'))
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--mode', choices=['wall','nsys'], default='wall')
a = p.parse_args()
cfg = json.loads((ROOT/'configs/5090/flashrt_nvfp4.json').read_text())
cfg['options']['calibration_cache'] = str(ROOT/'profile/flashrt-nvfp4-5090-20261030/reports/calibration-corrected')
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
  if True:
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
 metadata['scope']='GPU-ready complete aux + GPU state/noise -> GPU action; visual preparation excluded'
 metadata['collection_mode']=a.mode
 metadata['warmup_calls']=10
 if a.mode=='wall':
  ms=[]
  for i in range(50):
   torch.cuda.synchronize();t=time.perf_counter();y=frontend();torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
  metadata['timings']={'frontend':dict(samples_ms=ms,median_ms=statistics.median(ms),min_ms=min(ms),max_ms=max(ms))}
 else:
  torch.cuda.synchronize()
  torch.cuda.cudart().cudaProfilerStart()
  for i in range(10):
   torch.cuda.nvtx.range_push(f'REQUEST|frontend|{i}')
   y=frontend();torch.cuda.synchronize()
   torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()
 metadata['reference_error']=error(y,sample['reference_actions'])
 metadata['acceptance_threshold']=None
 (out/f'{a.mode}-metadata.json').write_text(json.dumps(metadata,indent=2))
 print('COMPLETE',a.mode,json.dumps(metadata.get('timings',{})),flush=True)
