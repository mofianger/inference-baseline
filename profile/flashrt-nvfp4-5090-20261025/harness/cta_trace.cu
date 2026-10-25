#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>
#include <string>
#include <mutex>
#include "nvbit_tool.h"
#include "nvbit.h"
struct Event {unsigned long long start,end; uint32_t sm,entries,exits,pad;};
constexpr unsigned STRIDE=12288, LAUNCHES=2048, CAP=STRIDE*LAUNCHES;
__managed__ Event* events=nullptr;
__managed__ unsigned long long count=0;
__managed__ int enabled=0;
struct Shape {unsigned gx,gy,gz,bx,by,bz;}; static std::vector<Shape> captured;
static bool ready=false;static uint32_t serial=0;static std::unordered_set<CUfunction> seen;static FILE* names=nullptr;
void nvbit_at_init(){setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC","1",1);}
void nvbit_tool_init(CUcontext ctx){if(!ready){cudaMallocManaged(&events,CAP*sizeof(Event));ready=true;}}
void instrument(CUcontext ctx,CUfunction f){
 if(!seen.insert(f).second)return;
 const auto& ins=nvbit_get_instrs(ctx,f);if(ins.empty())return;
 for(auto i:ins){bool first=i==ins.front(),last=std::string(i->getOpcodeShort())=="EXIT";if(!first&&!last)continue;
 nvbit_insert_call(i,"stamp",IPOINT_BEFORE);nvbit_add_call_arg_guard_pred_val(i);nvbit_add_call_arg_launch_val64(i,0);nvbit_add_call_arg_const_val64(i,(uint64_t)events);nvbit_add_call_arg_const_val64(i,(uint64_t)&count);nvbit_add_call_arg_const_val64(i,(uint64_t)&enabled);nvbit_add_call_arg_const_val32(i,last?1:0);
 }nvbit_enable_instrumented(ctx,f,true);
}
uint32_t tag(CUcontext ctx,CUfunction f){auto id=serial++;if(enabled&&names){fprintf(names,"%u\t%s\n",id,nvbit_get_func_name(ctx,f));fflush(names);}return id;}
void nvbit_at_graph_node_launch(CUcontext ctx,CUfunction f,CUstream stream,uint64_t handle){nvbit_set_at_launch(ctx,f,tag(ctx,f),stream,handle);}
void nvbit_at_cuda_event(CUcontext ctx,int is_exit,nvbit_api_cuda_t cbid,const char*,void* params,CUresult*){
 if(is_exit)return;
 CUfunction f=nullptr;CUstream stream=nullptr;Shape shape{};
 if(cbid==API_CUDA_cuLaunchKernel||cbid==API_CUDA_cuLaunchKernel_ptsz){auto p=(cuLaunchKernel_params*)params;f=p->f;stream=p->hStream;shape={p->gridDimX,p->gridDimY,p->gridDimZ,p->blockDimX,p->blockDimY,p->blockDimZ};}
 if(cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz){auto p=(cuLaunchKernelEx_params*)params;f=p->f;stream=p->config->hStream;auto c=p->config;shape={c->gridDimX,c->gridDimY,c->gridDimZ,c->blockDimX,c->blockDimY,c->blockDimZ};}
 if(f){cudaStreamCaptureStatus status;cudaStreamIsCapturing(stream,&status);if(status==cudaStreamCaptureStatusActive){captured.push_back(shape);instrument(ctx,f);nvbit_set_at_launch(ctx,f,tag(ctx,f));}}
}
extern "C" void cta_start(const char* path){cudaDeviceSynchronize();count=0;serial=0;cudaMemset(events,0,CAP*sizeof(Event));cudaDeviceSynchronize();names=fopen((std::string(path)+".names.tsv").c_str(),"w");FILE* cap=fopen((std::string(path)+".capture.tsv").c_str(),"w");for(auto s:captured)fprintf(cap,"%u\t%u\t%u\t%u\t%u\t%u\n",s.gx,s.gy,s.gz,s.bx,s.by,s.bz);fclose(cap);enabled=1;}
extern "C" void cta_stop(const char* path){cudaDeviceSynchronize();enabled=0;if(names){fclose(names);names=nullptr;}FILE* f=fopen((std::string(path)+".bin").c_str(),"wb");unsigned long long total=0;for(unsigned l=0;l<std::min(serial,LAUNCHES);l++)for(unsigned c=0;c<STRIDE;c++){Event& e=events[l*STRIDE+c];if(e.start){fwrite(&l,4,1,f);fwrite(&c,4,1,f);fwrite(&e,sizeof(Event),1,f);total++;}}fprintf(stderr,"CTA_RECORDS=%llu LAUNCHES=%u OVERFLOW=%llu\n",total,serial,count);fclose(f);fprintf(stderr,"CTA_EVENTS=%llu CAP=%u\n",count,CAP);}
