"""Pointer-compatible, preallocated F1-F8 controls; never modifies upstream."""
import torch
from extension import extension

DT={torch.uint8:0,torch.float16:5,torch.float32:6,torch.bfloat16:15}
NAMES={'fp8_residual_norm_quant','silu_mul_split_fp8_fp16','nvfp4_skinny_layer_norm',
       'nvfp4_skinny_sum_bias_gelu','nvfp4_skinny_sum_bias_residual_norm',
       'nvfp4_skinny_partials_attention','nvfp4_skinny_gemm_bf16_act'}

class Controls:
 def __init__(self,original):
  self.orig=original;self.ext=extension();self.buffers={};self.mode='fused';self.observe=False;self.records={}
 def buf(self,key,shape,dtype=torch.float32):
  key=(key,tuple(shape),dtype)
  if key not in self.buffers:self.buffers[key]=torch.zeros(shape,device='cuda',dtype=dtype)
  return self.buffers[key].data_ptr()
 def view(self,p,shape,dtype):return self.ext.view(int(p),list(shape),DT[dtype])
 def run(self,op,ptrs,r,n,eps=1e-5,stream=0):self.ext.run(op,[int(x) for x in ptrs],r,n,eps,stream)
 def __getattr__(self,name):
  orig=getattr(self.orig,name)
  if name not in NAMES:return orig
  def call(*args):
   if self.observe:self.record(name,args)
   if self.mode=='fused':return orig(*args)
   return self.split(name,args)
  return call
 def norm(self,h,sc,sh,out,sf,gs,bout,r,n,eps,st):
  tmp=self.buf('norm',[r,n]);self.run(0,[h,tmp],r,n,eps,st)
  if sc:self.run(1,[tmp,sc,sh],r,n,eps,st)
  if bout:self.run(11,[tmp,bout],r,n,eps,st)
  else:self.run(2,[tmp,out,sf,gs],r,n,eps,st)
 def split(self,name,a):
  if name=='fp8_residual_norm_quant':
   h,de,db,de2,copy,w,b,mode,eps,out,gs,bout,r,n,st=a
   if de or db or de2 or copy:
    self.orig.fp8_residual_norm_quant(h,de,db,de2,copy,w,b,mode,eps,0,gs,0,r,n,st)
   if out or bout:
    tmp=self.buf('half_norm',[r,n]);self.run(10,[h,w,b,mode,tmp,bout],r,n,eps,st)
    if out:self.run(7,[tmp,out,gs],r,n,eps,st)
  elif name=='silu_mul_split_fp8_fp16':
   g,u,y,total,gs,st=a;tmp=self.buf('silu',[total]);self.run(6,[g,u,tmp],1,total,stream=st);self.run(7,[tmp,y,gs],1,total,stream=st)
  elif name=='nvfp4_skinny_layer_norm':
   h,sc,sh,out,sf,gs,bout,r,n,eps,pdl,st=a;assert not pdl;self.norm(h,sc,sh,out,sf,gs,bout,r,n,eps,st)
  elif name=='nvfp4_skinny_sum_bias_gelu':
   part,splits,ag,wg,b,out,sf,gs,r,n,pdl,st=a;assert not pdl
   tmp=self.buf('sum',[r,n],torch.bfloat16);act=self.buf('gelu',[r,n])
   self.run(3,[part,splits,ag,wg,b,tmp],r,n,stream=st);self.run(5,[tmp,act],r,n,stream=st);self.run(2,[act,out,sf,gs],r,n,stream=st)
  elif name=='nvfp4_skinny_sum_bias_residual_norm':
   part,splits,ag,wg,b,h,sc,sh,out,sf,gs,bout,r,n,eps,pdl,st=a;assert not pdl
   tmp=self.buf('sum',[r,n],torch.bfloat16)
   self.run(3,[part,splits,ag,wg,b,tmp],r,n,eps,st);self.run(4,[tmp,h],r,n,eps,st)
   self.norm(h,sc,sh,out,sf,gs,bout,r,n,eps,st)
  elif name=='nvfp4_skinny_partials_attention':
   part,splits,ag,wg,b,n,k,v,ld,kv,static,out,r,heads,hd,scale,pdl,st=a;assert not pdl
   tmp=self.buf('attention_materialized',[r,n]);one=self.buf('one',[1]);self.view(one,[1],torch.float32).fill_(1) if not torch.cuda.is_current_stream_capturing() else None
   zero=self.buf('zero_bias',[n],torch.bfloat16)
   self.run(8,[part,splits,ag,wg,b,tmp],r,n,stream=st)
   self.orig.nvfp4_skinny_partials_attention(tmp,1,one,one,zero,n,k,v,ld,kv,static,out,r,heads,hd,scale,False,st)
  elif name=='nvfp4_skinny_gemm_bf16_act':
   x,gs,w,sfw,out,r,n,k,cfg,pdl,st=a;assert not pdl
   q=self.buf('actq',[r,k//2],torch.uint8);sf=self.buf('actsf',[r,k//16],torch.uint8)
   self.run(9,[x,q,sf,gs],r,k,stream=st)
   self.orig.nvfp4_skinny_gemm(q,sf,w,sfw,out,r,n,k,cfg,False,st)
  else:raise ValueError(name)
  return 0
 def record(self,name,a):
  """Snapshot representative real inputs and persistent output buffers before a call."""
  specs={};outs=[];mut=[];group=''
  def add(i,shape,dt,output=False,mutable=False):
   if a[i]:specs[i]=(shape,dt)
   if output and a[i]:outs.append(i)
   if mutable and a[i]:mut.append(i)
  bf=torch.bfloat16;f=torch.float32;h=torch.float16;u=torch.uint8
  if name=='fp8_residual_norm_quant':
   r,n=a[12:14];group=f'F1-{r}x{n}-norm{a[7]}-res{bool(a[1])}-ds{bool(a[3])}'
   if not a[9]:return
   for i in [0,1,3,4,11]:add(i,[r,n],h,i in [4,11],i==0)
   for i in [2,5,6]:add(i,[n],h)
   add(9,[r,n],u,True);add(10,[1],f);outs.append(0)
  elif name=='silu_mul_split_fp8_fp16':
   r,n=1,a[3];group='F2-SwiGLU'
   for i in [0,1]:add(i,[n],h)
   add(2,[n],u,True);add(4,[1],f)
  elif name=='nvfp4_skinny_layer_norm':
   r,n=a[7:9];group='F3-AdaLN'
   add(0,[r,n],bf)
   for i in [1,2]:add(i,[n],bf)
   add(3,[r,n//2],u,True);add(4,[r,n//16],u,True);add(5,[1],f);add(6,[r,n],bf,True)
  elif name=='nvfp4_skinny_sum_bias_gelu':
   r,n=a[8:10];group='F4-sum-GELU-quant';add(0,[a[1],r,n],f)
   for i in [2,3,7]:add(i,[1],f)
   add(4,[n],bf);add(5,[r,n//2],u,True);add(6,[r,n//16],u,True)
  elif name=='nvfp4_skinny_sum_bias_residual_norm':
   r,n=a[12:14];group='F6-output-AdaLN' if a[11] else 'F6-next-AdaLN' if a[6] else 'F5-residual-LN'
   add(0,[a[1],r,n],f)
   for i in [2,3,10]:add(i,[1],f)
   for i in [4,6,7]:add(i,[n],bf)
   add(5,[r,n],bf,True,True);add(8,[r,n//2],u,True);add(9,[r,n//16],u,True);add(11,[r,n],bf,True)
  elif name=='nvfp4_skinny_partials_attention':
   r,n=a[12],a[5];group=f'F7-attention-kv{a[9]}'
   add(0,[a[1],r,n],f);add(2,[1],f);add(3,[1],f);add(4,[n],bf)
   # Preserve row stride by cloning a full leading-stride span plus final head columns.
   for i in [6,7]:add(i,[(a[9]-1)*a[8]+a[13]*a[14]],bf)
   add(11,[r,a[13]*a[14]],bf,True)
  elif name=='nvfp4_skinny_gemm_bf16_act':
   r,n,k=a[5:8];group='F8-inline-quant';add(0,[r,k],bf);add(1,[1],f)
   add(2,[n*k//2],u);add(3,[n*k//16],u);add(4,[k//self.orig.nvfp4_skinny_config_k_chunk(a[8]),r,n],f,True)
  if group in self.records:return
  aa=list(a);buffers={}
  for i,(shape,dt) in specs.items():
   buffers[i]=self.view(a[i],shape,dt).clone();aa[i]=buffers[i].data_ptr()
  aa[-1]=0
  if name.startswith('nvfp4_'):aa[-2]=False
  self.records[group]=dict(group=group,name=name,args=aa,buffers=buffers,outputs=outs,mutations=mut,rows=r,n=n)
