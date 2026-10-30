"""Experimental segment replacement using pinned official FP8/NVFP4 kernels.

Generated functions only select original Python call sequences; CUDA kernels
are unchanged. Extra interface norms/quantization are retained and reported.
"""
import inspect,textwrap,types,torch
from flash_rt.models.groot_n17 import pipeline_thor as FP,pipeline_rtx_nvfp4 as NV
from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
def closure(fn):return dict(zip(fn.__code__.co_freevars,[c.cell_contents for c in fn.__closure__ or []]))
def fp8_segments():
 src=inspect.getsource(FP.dit_forward)
 a=src.index('        # ── AdaLN modulated norm1');b=src.index('        gemm.bf16_nn(O_ptr',a);c=src.index('        if "ff_proj_w_fp8_sm120"',b)
 chunks={'attention':src[a:b],'o':src[b:c],'ffn':src[c:]}
 result={}
 prefix='''def segment(gemm,fvk,bufs,weights,dims,attn,li,stream):
    Sa,D,FF=41,1536,6144
    Skv_text,Skv_image=dims['Skv_text'],dims['Skv_image']
    h_ptr,xn_ptr,o_out_ptr,ff_out_ptr=[int(bufs[k]) for k in ['h','xn','o_proj_out','ff_proj_out']]
    is_self=li%2==1
    j_attn=(li-1)//2 if is_self else li//2
    slots=attn.get_slot_ptrs('dit_self' if is_self else 'dit_cross',j_attn)
    Q_ptr,K_ptr,V_ptr,O_ptr=[slots[k] for k in ['Q','K','V','O']]
'''
 for key,chunk in chunks.items():
  ns={};code=prefix+textwrap.indent(textwrap.dedent(chunk),'    ');exec(compile(code,'<official-fp8-'+key+'>','exec'),ns);result[key]=ns['segment']
 return result
def nv_function():
 src=inspect.getsource(NV.dit_forward_nvfp4)
 src=src.replace('cfg: dict | None = None) -> None:', 'cfg: dict | None = None, hooks=None) -> None:')
 a=src.index('        # Attention straight');b=src.index('        # O projection',a);c=src.index('        # FFN up',b)
 prefix=src[:a];parts=[('attention',src[a:b]),('o',src[b:c]),('ffn',src[c:])]
 for key,part in parts:
  prefix+=f"        if hooks is not None: hooks.snapshot('{key}',li)\n        if hooks is not None and '{key}' in hooks.active:\n            hooks.run('{key}',li,stream)\n        else:\n"+textwrap.indent(part,'    ')
 ns=dict(vars(NV));exec(compile(prefix,'<official-nvfp4-segment-hooks>','exec'),ns);return ns['dit_forward_nvfp4'],prefix
def nv_local():
 src=inspect.getsource(NV.dit_forward_nvfp4)
 src=src.replace('cfg: dict | None = None) -> None:', "cfg: dict | None = None, layer=0, segment='attention') -> None:")
 init=src.index('    _ck(fvk.nvfp4_skinny_layer_norm');loop=src.index('    for li in range(LAYERS):');a=src.index('        # Attention straight');b=src.index('        # O projection',a);c=src.index('        # FFN up',b)
 code=src[:init]+"    if segment=='attention':\n"+textwrap.indent(src[init:loop].replace('scale[0]','scale[layer]').replace('shift[0]','shift[layer]'),'    ')+src[loop:a].replace('range(LAYERS)','[layer]')
 for key,part in [('attention',src[a:b]),('o',src[b:c]),('ffn',src[c:])]:code+=f"        if segment=='{key}':\n"+textwrap.indent(part,'    ')
 ns=dict(vars(NV));exec(compile(code,'<nvfp4-common-boundary-segment>','exec'),ns);return ns['dit_forward_nvfp4']
class Hybrid:
 def __init__(self,f,ctl,out):
  self.f,self.ctl=f,ctl;self.active=set();self.observe=False;self.snapshots={};self.fp8=fp8_segments();self.local_nv=nv_local();self.nv,src=nv_function();(out/'generated-nvfp4.py').write_text(src)
  state,step=f._kdit_fwd;self.state=state;cc=closure(step)
  self.ae,self.post,self.weights,self.bp,self.dims=[cc[k] for k in ['_ae_fwd','_post_fwd','step_weights','bp','dims']]
  # Calibrate with the official routine, then frozen previous-run scalars are
  # applied by setup.scales before any experimental graph is captured.
  f._calibrate_quantize_dit_ffn(4,40,state,self.ae,self.post,self.weights,self.bp,self.dims)
  self.native_graph=f._k_dit_graph
  self.original_kv=(f._dit_cross_K,f._dit_cross_V)
  self.contiguous_kv=([k.contiguous() for k in f._dit_cross_K],[v.contiguous() for v in f._dit_cross_V])
  self.prepare_split_cross()
 def snapshot(self,part,li):
  if not self.observe or li not in [0,1,2,31]:return
  f=self.f;key=(self.step,li,part)
  rec={'h':f._infer_bufs['dit_h'].clone()}
  if part=='o':
   slots=f._dit_attn.get_slot_ptrs('dit_self' if li%2 else 'dit_cross',(li-1)//2 if li%2 else li//2)
   rec['o']=self.ctl.view(slots['O'],[41,1536],torch.bfloat16).clone()
  if part=='ffn':
   f._fvk.layer_norm_no_affine_bf16(self.bp['h'],self.bp['xn'],41,1536,1e-5,torch.cuda.current_stream().cuda_stream)
   rec['xn']=f._infer_bufs['dit_xn'].clone()
  self.snapshots[key]=rec
 def local(self,part,li,step,precision):
  f=self.f;K=f._fvk;s=torch.cuda.current_stream().cuda_stream;w=self.weights[step];self.step=step
  if precision=='FP8':
   self.fp8[part](f._gemm,K,self.bp,w,self.dims,f._dit_attn,li,s)
   if part=='ffn':
    sc=w['scale_msa'][li+1] if li<31 else w['out_scale'];sh=w['shift_msa'][li+1] if li<31 else w['out_shift']
    K.ada_layer_norm_bf16(self.bp['h'],sc,sh,self.bp['xn'],41,1536,1e-5,s)
  else:
   # The terminal fused consumer writes a BF16 result at the shared boundary.
   class LocalK:
    def __getattr__(obj,name):return getattr(K,name)
    def nvfp4_skinny_sum_bias_residual_norm(obj,*args):
     aa=list(args);aa[8]=0;aa[9]=0;aa[10]=0;aa[11]=self.bp['xn'];return K.nvfp4_skinny_sum_bias_residual_norm(*aa)
   if part=='ffn':
    bb=f._dit_nvfp4_bufs;self.ctl.run(9,[self.bp['xn'],bb['xn'],bb['xn_sf'],bb['one']],41,1536,stream=s)
   self.local_nv(LocalK(),f._dit_nvfp4_bufs,w,dict(self.dims,kv_ld=f._dit_cross_K[0].stride(0)),attn=f._dit_attn,stream=s,pdl=False,layer=li,segment=part)
  if part=='attention':
   slots=f._dit_attn.get_slot_ptrs('dit_self' if li%2 else 'dit_cross',(li-1)//2 if li%2 else li//2)
   return self.ctl.view(slots['O'],[41,1536],torch.bfloat16)
  return f._infer_bufs['dit_xn']
 def set_kv(self,contiguous):
  f=self.f;k,v=self.contiguous_kv if contiguous else self.original_kv
  f._dit_cross_K,f._dit_cross_V=k,v;f._dit_attn.dit_cross_K=[t.view(t.shape[0],32,48) for t in k];f._dit_attn.dit_cross_V=[t.view(t.shape[0],32,48) for t in v]
 def prepare_split_cross(self):
  f=self.f;K=f._fvk;self.split_sites=[]
  for name,g in f._ck_nvfp4['groups'].items():
   wg=(g['alpha']/g['a_gs']).clone()
   for slot,j in enumerate(g['blocks']):
    for off,kv in enumerate(['k','v']):
     w=f._ckpt_bf16_tensor(f'action_head.model.transformer_blocks.{2*j}.attn1.to_{kv}.weight').contiguous()
     q=torch.empty(1536,1024,device='cuda',dtype=torch.uint8);sf=torch.empty(1536,128,device='cuda',dtype=torch.uint8)
     K.nvfp4_skinny_quantize_weight(w.data_ptr(),wg.data_ptr(),q.data_ptr(),sf.data_ptr(),1536,2048,0,f._DIT_WEIGHT_MSE_STEPS)
     sw=torch.zeros(K.nvfp4_sf_swizzled_bytes(1536,2048),device='cuda',dtype=torch.uint8);K.nvfp4_sf_linear_to_swizzled(sf.data_ptr(),sw.data_ptr(),1536,2048,True,0)
     bias=g['b'][(2*slot+off)*1536:(2*slot+off+1)*1536]
     self.split_sites.append((g,j,off,q,sw,bias,wg))
 def cross(self,mode,s):
  f=self.f;K=f._fvk
  if mode=='cross-fp8':GrootN17TorchFrontendThor._cross_kv_fwd(f,s);return
  if mode!='cross-split':f._cross_kv_fwd(s);return
  K.cast_fp16_to_bf16(f._ck_bb_src.data_ptr(),f._ck_bb.data_ptr(),f.Se*2048,s)
  for name,g in f._ck_nvfp4['groups'].items():
   idx=f._ck_text_idx if name=='text' else f._ck_image_idx
   K.embedding_lookup_bf16(idx.data_ptr(),f._ck_bb.data_ptr(),g['src'].data_ptr(),g['rows'],2048,s)
   K.nvfp4_quant(g['src'].data_ptr(),2048,g['a'].data_ptr(),g['sfa'].data_ptr(),g['a_gs'].data_ptr(),g['rows'],2048,s)
  for g,j,off,q,sf,b,wg in self.split_sites:
   dest=(f._dit_cross_K if off==0 else f._dit_cross_V)[j]
   K.nvfp4_gemm_bias_res_bf16out(g['a'].data_ptr(),q.data_ptr(),g['sfa'].data_ptr(),sf.data_ptr(),b.data_ptr(),0,dest.data_ptr(),g['rows'],1536,2048,g['alpha'].data_ptr(),0.,0,s)
 def next_norm(self,li,s,bf16=False):
  f=self.f;K=f._fvk;w=self.weights[self.step];bb=f._dit_nvfp4_bufs
  scale=w['scale_msa'][li+1] if li<31 else w['out_scale'];shift=w['shift_msa'][li+1] if li<31 else w['out_shift']
  out=bb['hmod'] if li==31 else self.bp['xn'] if bf16 else 0
  K.nvfp4_skinny_layer_norm(bb['h'],scale,shift,0 if out else bb['xn'],0 if out else bb['xn_sf'],bb['one'],out,41,1536,1e-5,False,s)
 def run(self,part,li,s):
  f=self.f;K=f._fvk;w=self.weights[self.step];bb=f._dit_nvfp4_bufs
  if part=='ffn':K.layer_norm_no_affine_bf16(self.bp['h'],self.bp['xn'],41,1536,1e-5,s)
  self.fp8[part](f._gemm,K,self.bp,w,self.dims,f._dit_attn,li,s)
  if part=='o':self.ctl.run(9,[self.bp['xn'],bb['xn'],bb['xn_sf'],bb['one']],41,1536,stream=s)
  if part=='ffn':self.next_norm(li,s)
 def dit(self,step,s):
  self.step=step
  dims=dict(self.dims,kv_ld=self.f._dit_cross_K[0].stride(0))
  self.nv(self.f._fvk,self.f._dit_nvfp4_bufs,self.weights[step],dims,attn=self.f._dit_attn,stream=s,pdl=False,hooks=self)
 def all(self,mode,s):
  self.cross(mode,s);self.state(s)
  for step in range(4):self.ae(step,s);self.dit(step,s);self.post(step,s,out_norm_done=True)
