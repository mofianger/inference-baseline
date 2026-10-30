"""Freeze calibration scalars across official backbone variants before graph capture."""
import os,json
from pathlib import Path
def install(root):
 import torch
 from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
 from flash_rt.frontends.torch.groot_n17_rtx_fp4 import GrootN17TorchFrontendRtxFP4
 root.mkdir(parents=True,exist_ok=True)
 def freeze(name,values):
  path=root/(name+'.json')
  actual={k:float(t.item()) for k,t in values.items()}
  if not path.exists():
   if not os.environ['RECIPE'].endswith('-fused'):raise RuntimeError('Fused calibration must run first')
   path.write_text(json.dumps(actual,indent=2))
  saved=json.loads(path.read_text());assert actual.keys()==saved.keys()
  for k,t in values.items():t.fill_(saved[k])
 orig=GrootN17TorchFrontendThor._calibrate_quantize_dit_ffn
 def calibrated(self,*args,**kwargs):
  orig(self,*args,**kwargs)
  freeze('fp8-dit',{str(i):t for i,t in enumerate(self._dit_ff_fp8_keep) if t.dtype==torch.float32 and t.numel()==1})
 GrootN17TorchFrontendThor._calibrate_quantize_dit_ffn=calibrated
 orig4=GrootN17TorchFrontendRtxFP4._setup_cross_kv_nvfp4
 def cross(self,*args,**kwargs):
  orig4(self,*args,**kwargs)
  freeze('nvfp4-cross',{name+'/'+key:g[key] for name,g in self._ck_nvfp4['groups'].items() for key in ['a_gs','alpha']})
  self._cross_kv_fwd(0)
 GrootN17TorchFrontendRtxFP4._setup_cross_kv_nvfp4=cross
