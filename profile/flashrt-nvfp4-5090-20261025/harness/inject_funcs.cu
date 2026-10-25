#include <stdint.h>
struct Event {uint64_t t; uint32_t launch,cta,sm,warp,exit,pad;};
constexpr unsigned CAP=8000000;
extern "C" __device__ __noinline__ void stamp(int guard,uint64_t id,uint64_t evp,uint64_t cp,uint64_t ep,int end){
 if(!guard || !*(int*)ep)return;
 unsigned lane;asm("mov.u32 %0, %%laneid;":"=r"(lane));unsigned mask=__activemask();if(lane!=(unsigned)(__ffs(mask)-1))return;
 unsigned sm;unsigned long long t;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));
 unsigned tid=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
 auto idx=atomicAdd((unsigned long long*)cp,1ULL);if(idx<CAP)((Event*)evp)[idx]={t,(uint32_t)id,blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z),sm,tid/32,(unsigned)end,0};
}
