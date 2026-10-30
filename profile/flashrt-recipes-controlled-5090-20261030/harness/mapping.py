"""Independent eager source mapping. Synchronization here is NOT benchmark timing."""
import inspect,json,types,linecache
def capture(f,sample,aux,out):
 import torch
 from flash_rt import flash_rt_kernels as K,flash_rt_fa2 as FA
 records=[];phase=['Backbone'];original=[]
 def wrap(fn,name):
  def call(*args,**kwargs):
   frames=[];frame=inspect.currentframe().f_back
   while frame:
    path=frame.f_code.co_filename
    if '/flash_rt/' in path:
     local=frame.f_locals
     scalar={k:v for k,v in local.items() if isinstance(v,(int,float,str,bool)) and (k in ['li','i','j','step','S','Sv','Se','Sa','D','FF','N','K','M','NH','HD','NHQ','NHKV','Skv_text','Skv_image','layers_subset','dit_fp8_impl'] or k.startswith('a_'))}
     frames.append(dict(file=path.split('/FlashRT/')[-1],line=frame.f_lineno,function=frame.f_code.co_name,locals=scalar,code=''.join(linecache.getlines(path)[max(0,frame.f_lineno-2):frame.f_lineno+4])))
    frame=frame.f_back
   idx=len(records);rec=dict(id=idx,api=name,phase=phase[0],frames=frames,signature=(fn.__doc__ or '')[:6000],args=[v if isinstance(v,(int,float,str,bool)) else str(type(v)) for v in args],kwargs={k:v if isinstance(v,(int,float,str,bool)) else str(type(v)) for k,v in kwargs.items()})
   records.append(rec)
   torch.cuda.synchronize();torch.cuda.nvtx.range_push('OP|'+str(idx))
   try:result=fn(*args,**kwargs);torch.cuda.synchronize()
   finally:torch.cuda.nvtx.range_pop()
   return result
  return call
 for module in [K,FA]:
  for name in dir(module):
   fn=getattr(module,name)
   if inspect.isbuiltin(fn):
    original.append((module,name,fn));setattr(module,name,wrap(fn,name))
 class Proxy:
  def __init__(self,obj):self.obj=obj;self.cache={}
  def __getattr__(self,name):
   if name not in self.cache:self.cache[name]=wrap(getattr(self.obj,name),'GemmRunner.'+name)
   return self.cache[name]
 proxies={}
 def proxy_for(obj):
  if id(obj) not in proxies:proxies[id(obj)]=Proxy(obj)
  return proxies[id(obj)]
 for name,value in list(vars(f).items()):
  if type(value).__name__=='GemmRunner':setattr(f,name,proxy_for(value))
 for name in ['index_copy_','zero_']:
  fn=getattr(torch.Tensor,name);original.append((torch.Tensor,name,fn));setattr(torch.Tensor,name,wrap(fn,'Tensor.'+name))
 seen=set()
 def patch_closure(fn):
  if not isinstance(fn,types.FunctionType) or id(fn) in seen:return
  seen.add(id(fn))
  for cell in fn.__closure__ or []:
   try:v=cell.cell_contents
   except ValueError:continue
   if type(v).__name__=='GemmRunner':cell.cell_contents=proxy_for(v)
   elif isinstance(v,types.FunctionType):patch_closure(v)
 for fn in [f._kbb_forward,*f._kdit_fwd]:patch_closure(fn)
 f._kbb_vit_h.copy_(aux['pixel_features'].half().reshape(f._S_vit,1024))
 f._kbb_llm_h.copy_(aux['llm_input_embeds'].half().reshape(f.Se,2048))
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
 f._kbb_forward(0)
 phase[0]='CrossKV';f._ck_bb_src.copy_(f._kbb_vlsa_h.reshape(f.Se,2048).half());f._cross_kv_fwd(0)
 phase[0]='State';f._k_state_in.copy_(sample['inputs']['state'].reshape(1,132).bfloat16());f._k_actions.copy_(sample['noise'].reshape(40,132).bfloat16())
 state,step=f._kdit_fwd;state(0)
 for i in range(4):phase[0]='Step '+str(i+1);step(i,0)
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 (out/'operations.json').write_text(json.dumps(records,indent=2))
 for module,name,fn in original:setattr(module,name,fn)
 # Process exits after one normal graph replay; the captured graph is unaffected.
