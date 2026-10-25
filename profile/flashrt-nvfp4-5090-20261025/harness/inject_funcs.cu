#include <stdint.h>
struct Event {unsigned long long start,end; uint32_t sm,entries,exits,pad;};
constexpr unsigned STRIDE=12288, LAUNCHES=2048;
extern "C" __device__ __noinline__ void stamp(int guard,uint64_t id,uint64_t evp,uint64_t cp,uint64_t ep,int end){
 if(!guard || !*(int*)ep)return;
 unsigned lane;asm("mov.u32 %0, %%laneid;":"=r"(lane));unsigned mask=__activemask();if(lane!=(unsigned)(__ffs(mask)-1))return;
 unsigned sm;unsigned long long t;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));
 unsigned cta=blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z);
 if(id>=LAUNCHES||cta>=STRIDE){atomicAdd((unsigned long long*)cp,1ULL);return;}
 Event* e=(Event*)evp+id*STRIDE+cta;
 if(!end){atomicCAS(&e->start,0ULL,t);atomicMin(&e->start,t);e->sm=sm;atomicAdd(&e->entries,(unsigned)__popc(mask));}
 else{atomicMax(&e->end,t);atomicAdd(&e->exits,(unsigned)__popc(mask));}
}
