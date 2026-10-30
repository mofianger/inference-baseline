import argparse,os
from pathlib import Path
os.environ['FLASHRT_GROOT_N17_DIT_PDL']='0'
from setup import torch
from flash_rt import flash_rt_kernels as K
from controls import Controls
from measure import graph
p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);a=p.parse_args()
with torch.inference_mode():
 ctl=Controls(K);fixtures=torch.load(a.fixtures,weights_only=False);tasks=[]
 for key,r in fixtures.items():
  args=list(r['args']);buf={i:t.cuda() for i,t in r['buffers'].items()}
  for i,t in buf.items():args[i]=t.data_ptr()
  copies={i:buf[i].clone() for i in r['mutations']}
  for mode in ['fused','unfused']:
   def fn(args=args,name=r['name'],mode=mode):
    aa=[*args[:-1],torch.cuda.current_stream().cuda_stream]
    if mode=='fused':getattr(K,name)(*aa)
    else:ctl.split(name,aa)
   tasks.append((key,mode,graph(fn),buf,copies))
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
 for key,mode,g,buf,copies in tasks:
  for i in range(5):
   for j,x in copies.items():buf[j].copy_(x)
   torch.cuda.synchronize();torch.cuda.nvtx.range_push(f'LOCAL|{key}|{mode}|{i}')
   g.replay();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
 torch.cuda.cudart().cudaProfilerStop()
