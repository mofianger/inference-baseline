"""Same-task NVFP4/FP8/BF16 controls, with quantization inside measured regions."""
import os,argparse,json
from pathlib import Path
os.environ['FLASHRT_GROOT_N17_DIT_PDL']='0'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:8]))
from setup import setup,error,torch
from controls import Controls
from measure import paired_graphs
from flash_rt.models.groot_n17.pipeline_rtx_nvfp4 import SKINNY_CFG
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--fixtures',required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
ptr=lambda t:t.data_ptr()
stream=lambda:torch.cuda.current_stream().cuda_stream

class Linear:
 def __init__(self,f,ctl,x,w,b,cross=None):
  self.f,self.ctl,self.x,self.w,self.b,self.cross=f,ctl,x,w,b,cross
  self.m,self.k=x.shape;self.n=w.shape[0];self.mp=(self.m+15)//16*16
  self.y_pad=torch.empty(self.mp,self.n,device='cuda',dtype=torch.bfloat16);self.y=self.y_pad[:self.m]
  self.a8_pad=torch.zeros(self.mp,self.k,device='cuda',dtype=torch.float8_e4m3fn);self.a8=self.a8_pad[:self.m]
  self.as8=(x.float().abs().amax()/448).clamp_min(1e-12).reshape(1)
  self.ws8=(w.float().abs().amax()/448).clamp_min(1e-12).reshape(1)
  self.w8col=(w.float()/self.ws8).to(torch.float8_e4m3fn).t();self.w8=self.w8col.contiguous()
  self.alpha=float((self.as8*self.ws8).item())
  self.gemm=f._gemm
  self.fp8_bias=True
  f._fvk.quantize_fp8_static(ptr(x),ptr(self.a8),ptr(self.as8),x.numel(),stream())
  try:self.gemm.fp8_nn_bias_bf16(ptr(self.a8),ptr(self.w8),ptr(self.y),ptr(b),self.m,self.n,self.k,self.alpha,stream())
  except RuntimeError:self.fp8_bias=False
  self.scaled_mm=False
  try:
   torch.ops.aten._scaled_mm.out(self.a8_pad,self.w8col,self.as8,self.ws8,bias=b,out_dtype=torch.bfloat16,out=self.y_pad)
   self.scaled_mm=True
  except RuntimeError:pass
  self.fp8_backend='torch_scaled_mm_padded' if self.scaled_mm else 'cublasLt_bias' if self.fp8_bias else 'cublasLt_bf16_plus_bias'
  self.half=torch.empty_like(self.y,dtype=torch.float16)
  self.w4,self.sf4,self.wgs,self.bias4=f._quantize_nvfp4_site(w,b)
  self.one=torch.ones(1,device='cuda');self.a4=torch.empty(self.m,self.k//2,device='cuda',dtype=torch.uint8);self.as4=torch.empty(self.m,self.k//16,device='cuda',dtype=torch.uint8)
  self.cfg=SKINNY_CFG.get((self.n,self.k),11)
  self.splits=self.k//f._fvk.nvfp4_skinny_config_k_chunk(self.cfg)
  self.part=torch.empty(self.splits,self.m,self.n,device='cuda')
  self.keep=[]
 def nv_gemm(self,fused=True,x=None):
  x=self.x if x is None else x
  K=self.f._fvk
  if self.cross is not None:
   g=self.cross
   K.nvfp4_quant(ptr(x),self.k,ptr(g['a']),ptr(g['sfa']),ptr(g['a_gs']),self.m,self.k,stream())
   K.nvfp4_gemm_bias_res_bf16out(ptr(g['a']),ptr(g['w']),ptr(g['sfa']),ptr(g['sfb']),ptr(g['b']),0,ptr(self.y),self.m,self.n,self.k,ptr(g['alpha']),0.,0,stream())
   return
  if fused:K.nvfp4_skinny_gemm_bf16_act(ptr(x),ptr(self.one),self.w4,self.sf4,ptr(self.part),self.m,self.n,self.k,self.cfg,False,stream())
  else:
   self.ctl.run(9,[ptr(x),ptr(self.a4),ptr(self.as4),ptr(self.one)],self.m,self.k,stream=stream())
   K.nvfp4_skinny_gemm(ptr(self.a4),ptr(self.as4),self.w4,self.sf4,ptr(self.part),self.m,self.n,self.k,self.cfg,False,stream())
 def call(self,precision,fused=True,x=None):
  x=self.x if x is None else x
  if precision=='BF16':
   if fused:torch.addmm(self.b,x,self.w.t(),out=self.y)
   else:torch.mm(x,self.w.t(),out=self.y);self.y.add_(self.b)
  elif precision=='FP8':
   self.f._fvk.quantize_fp8_static(ptr(x),ptr(self.a8),ptr(self.as8),x.numel(),stream())
   if self.scaled_mm:
    torch.ops.aten._scaled_mm.out(self.a8_pad,self.w8col,self.as8,self.ws8,bias=self.b if fused else None,out_dtype=torch.bfloat16,out=self.y_pad)
    if not fused:self.y.add_(self.b)
   elif fused and self.fp8_bias:self.gemm.fp8_nn_bias_bf16(ptr(self.a8),ptr(self.w8),ptr(self.y),ptr(self.b),self.m,self.n,self.k,self.alpha,stream())
   else:
    self.f._fvk.fp8_gemm_descale_bf16out(ptr(self.a8),ptr(self.w8),ptr(self.y),self.m,self.n,self.k,ptr(self.as8),ptr(self.ws8),stream())
    self.y.add_(self.b)
  else:
   self.nv_gemm(fused,x)
   if self.cross is None:self.ctl.run(3,[ptr(self.part),self.splits,ptr(self.one),self.wgs,self.bias4,ptr(self.y)],self.m,self.n,stream=stream())
  return self.y

with torch.inference_mode():
 adapter,sample,aux,fn,meta=setup();f=adapter.frontend;K=f._fvk;ctl=Controls(K)
 fixtures=torch.load(a.fixtures,weights_only=False)
 def restore(key):
  r=fixtures[key];args=list(r['args']);bb={i:t.cuda() for i,t in r['buffers'].items()}
  for i,t in bb.items():args[i]=ptr(t)
  return args,bb
 def norm_input(key):
  aa,bb=restore(key);ctl.split(fixtures[key]['name'],aa)
  return ctl.view(ctl.buf('norm',[41,1536]),[41,1536],torch.float32).bfloat16().clone()
 xq=norm_input('F3-AdaLN');xqkv=norm_input('F6-next-AdaLN');xup=norm_input('F5-residual-LN')
 aa,bb=restore('F4-sum-GELU-quant');ctl.split(fixtures['F4-sum-GELU-quant']['name'],aa)
 xdn=ctl.view(ctl.buf('gelu',[41,6144]),[41,6144],torch.float32).bfloat16().clone()
 _,bb=restore('F8-inline-quant');xo=bb[0].clone()
 def weight(li,name):return f._ckpt_bf16_tensor(f'action_head.model.transformer_blocks.{li}.{name}')
 specs=[('Q1-crossQ',xq,0,'attn1.to_q'),('Q2-selfQKV',xqkv,1,'qkv'),('Q3-O',xo,0,'attn1.to_out.0'),('Q4-FFup',xup,0,'ff.net.0.proj'),('Q5-FFdown',xdn,0,'ff.net.2')]
 tasks={}
 for label,x,li,name in specs:
  if name=='qkv':w=torch.cat([weight(li,f'attn1.to_{s}.weight') for s in 'qkv']);b=torch.cat([weight(li,f'attn1.to_{s}.bias') for s in 'qkv'])
  else:w=weight(li,name+'.weight');b=weight(li,name+'.bias')
  tasks[label]=Linear(f,ctl,x,w,b)
 fn()
 for name,g in f._ck_nvfp4['groups'].items():
  w=torch.cat([weight(2*j,f'attn1.to_{kv}.weight') for j in g['blocks'] for kv in ['k','v']]);b=g['b'].clone()
  tasks['Q6-crossKV-text' if name=='text' else 'Q7-crossKV-image']=Linear(f,ctl,g['src'].clone(),w,b,g)
 results=[];trace_tasks={}
 for label,L in tasks.items():
  ref=(L.x.float()@L.w.float().t()+L.b.float()).bfloat16()
  funcs={};errors={}
  variants=[(p,z) for p in ['BF16','FP8','NVFP4'] for z in [False,True] if not (L.cross is not None and p=='NVFP4' and not z)]
  for precision,fused in variants:
   key=precision+('/fused' if fused else '/split')
   funcs[key]=lambda L=L,p=precision,z=fused:L.call(p,z)
   y=funcs[key]();torch.cuda.synchronize();errors[key]=error(y,ref)
   assert torch.isfinite(y).all(),(label,key)
  timing=paired_graphs(funcs)
  results.append(dict(task=label,shape=[L.m,L.n,L.k],boundary='BF16 activation -> BF16 complete linear+bias; runtime activation quantization included',errors=errors,timing=timing,fp8_bias_supported=L.fp8_bias,fp8_backend=L.fp8_backend,fp8_rows=L.mp if L.scaled_mm else L.m))
  trace_tasks[label]=funcs
  print('LINEAR',label,json.dumps({k:v['median_us'] for k,v in timing['warm'].items()}),flush=True)
 # Q8: complete two-linear FFN, with real FlashRT fused NVFP4 GELU+quant consumer.
 U,D=tasks['Q4-FFup'],tasks['Q5-FFdown'];act=torch.empty_like(U.y);tmp=ctl.buf('ffn_float',[41,6144]);buf4=torch.empty_like(U.y,dtype=torch.uint8)[:,:3072].contiguous();sf4=torch.empty(41,384,device='cuda',dtype=torch.uint8)
 def ffn(p,z):
  if p=='NVFP4' and z:
   U.nv_gemm(True)
   K.nvfp4_skinny_sum_bias_gelu(ptr(U.part),U.splits,ptr(U.one),U.wgs,U.bias4,ptr(buf4),ptr(sf4),ptr(U.one),41,6144,False,stream())
   K.nvfp4_skinny_gemm(ptr(buf4),ptr(sf4),D.w4,D.sf4,ptr(D.part),41,1536,6144,D.cfg,False,stream())
   ctl.run(3,[ptr(D.part),D.splits,ptr(D.one),D.wgs,D.bias4,ptr(D.y)],41,1536,stream=stream())
  else:
   U.call(p,z);ctl.run(5,[ptr(U.y),tmp],41,6144,stream=stream());ctl.run(11,[tmp,ptr(act)],41,6144,stream=stream());D.call(p,z,act)
  return D.y
 funcs={p+('/fused' if z else '/split'):(lambda p=p,z=z:ffn(p,z)) for p in ['BF16','FP8','NVFP4'] for z in [False,True]}
 ref=ffn('BF16',True).clone();errs={k:error(fun(),ref) for k,fun in funcs.items()};timing=paired_graphs(funcs)
 results.append(dict(task='Q8-complete-FFN',shape=[41,1536,6144],boundary='BF16 input -> BF16 FFN output; GELU intermediate precision follows each implementation',errors=errs,timing=timing))
 trace_tasks['Q8-complete-FFN']=funcs
 # Q9: O projection + BF16 residual + LayerNorm, common BF16 output.
 _,rb=restore('F5-residual-LN');res=rb[5].clone();h=torch.empty_like(res);yn=torch.empty_like(res);normtmp=ctl.buf('q9norm',[41,1536]);L=tasks['Q3-O']
 def q9(p,z,L=L):
  h.copy_(res)
  if p=='NVFP4' and z:
   L.nv_gemm(True);K.nvfp4_skinny_sum_bias_residual_norm(ptr(L.part),L.splits,ptr(L.one),L.wgs,L.bias4,ptr(h),0,0,0,0,0,ptr(yn),41,1536,1e-5,False,stream())
  else:
   L.call(p,z);h.add_(L.y);ctl.run(0,[ptr(h),normtmp],41,1536,stream=stream());ctl.run(11,[normtmp,ptr(yn)],41,1536,stream=stream())
  return yn
 funcs={p+('/fused' if z else '/split'):(lambda p=p,z=z:q9(p,z)) for p in ['BF16','FP8','NVFP4'] for z in [False,True]}
 ref=q9('BF16',True).clone();errs={k:error(fun(),ref) for k,fun in funcs.items()};timing=paired_graphs(funcs)
 results.append(dict(task='Q9-O-residual-LN',shape=[41,1536,1536],boundary='BF16 attention output + fixed residual -> BF16 LayerNorm output; equal residual copy included',errors=errs,timing=timing));trace_tasks['Q9-O-residual-LN']=funcs
 # Q10: projection + attention; FlashRT direct partials vs efficient PyTorch SDPA.
 for kv in [25,192,41]:
  aa,bb=restore(f'F7-attention-kv{kv}');L=tasks['Q2-selfQKV'] if kv==41 else tasks['Q1-crossQ']
  # Image-cross layer uses its own projection weight, rather than reusing text Q weights.
  if kv==192:L=Linear(f,ctl,L.x,weight(2,'attn1.to_q.weight'),weight(2,'attn1.to_q.bias'))
  o=torch.empty(41,32,48,device='cuda',dtype=torch.bfloat16)
  if kv!=41:
   k=bb[6].as_strided((kv,32,48),(aa[8],48,1));v=bb[7].as_strided((kv,32,48),(aa[8],48,1))
  def attention(p,L=L,kv=kv,aa=aa,bb=bb,o=o,k=None if kv==41 else k,v=None if kv==41 else v):
   if p=='NVFP4':
    L.nv_gemm(True)
    K.nvfp4_skinny_partials_attention(ptr(L.part),L.splits,ptr(L.one),L.wgs,L.bias4,L.n,0 if kv==41 else ptr(bb[6]),0 if kv==41 else ptr(bb[7]),aa[8],kv,kv!=41,ptr(o),41,32,48,48**-.5,False,stream())
    return o
   L.call(p,True)
   if kv==41:q,k,v=L.y.reshape(41,3,32,48).unbind(1)
   else:q=L.y.reshape(41,32,48)
   y=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0))
   o.copy_(y.squeeze(0).transpose(0,1));return o
  funcs={p:(lambda p=p,fn=attention:fn(p)) for p in ['BF16','FP8','NVFP4']};ref=funcs['BF16']().clone();errs={k:error(fun(),ref) for k,fun in funcs.items()};timing=paired_graphs(funcs)
  label=f'Q10-projection-attention-kv{kv}';results.append(dict(task=label,boundary='BF16 input -> BF16 attention output; external KV fixed; implementation-specific fusion allowed',errors=errs,timing=timing));trace_tasks[label]=funcs
 (out/'precision.json').write_text(json.dumps(dict(metadata=meta,results=results,notes=['Static FP8 activation scale determined before timing from the frozen activation. NVFP4 uses dynamic per-16 scales.', 'Real intermediate activations reconstructed from captured arithmetic and rounded to the common BF16 input interface.', 'Each variant uses its native high-performance implementation; differences are task-level, not pure bit-width isolation.']),indent=2))
 if a.trace:
  gs={label:{k:__import__('measure').graph(fn) for k,fn in funcs.items()} for label,funcs in trace_tasks.items()}
  torch.cuda.cudart().cudaProfilerStart()
  for label,variants in gs.items():
   for key,g in variants.items():
    for i in range(3):
     torch.cuda.nvtx.range_push(f'PRECISION|{label}|{key}|{i}');g.replay();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()
 print('PRECISION COMPLETE',len(results),flush=True)
