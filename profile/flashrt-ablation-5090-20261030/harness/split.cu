// Experimental unfused controls. Arithmetic helpers are from the pinned FlashRT source.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include "nvfp4_skinny_common.cuh"
using namespace flash_rt::nvfp4_skinny;
using bf=__nv_bfloat16;
__device__ float bsum(float x,float* red){x=warp_sum(x);int lane=threadIdx.x&31,wi=threadIdx.x>>5;__syncthreads();if(!lane)red[wi]=x;__syncthreads();return warp_sum(lane<(blockDim.x>>5)?red[lane]:0.f);}
__global__ void norm_bf(const bf*x,float*y,int n,float eps){
 __shared__ float red[32];int c=threadIdx.x*4,r=blockIdx.x;float v[4];
 for(int j=0;j<4;j++)v[j]=__bfloat162float(x[r*n+c+j]);
 float mean=bsum(v[0]+v[1]+v[2]+v[3],red)/n;
 for(int j=0;j<4;j++)v[j]-=mean;
 float inv=rsqrtf(bsum(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3],red)/n+eps);
 for(int j=0;j<4;j++)y[r*n+c+j]=v[j]*inv;
}
__global__ void mod(float*x,const bf*s,const bf*b,int total,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)x[i]=x[i]*(1.f+__bfloat162float(s[i%n]))+__bfloat162float(b[i%n]);}
template<bool BF> __global__ void quant4(const void*x,unsigned char*y,unsigned char*sf,const float*gs,int total){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=total/16)return;float v[16];
 for(int j=0;j<16;j++)v[j]=(BF?__bfloat162float(((const bf*)x)[b*16+j]):((const float*)x)[b*16+j])/(*gs);
 quant_block16(v,y+b*8,sf+b);
}
__global__ void reduce_bias(const float*p,int splits,const float*ag,const float*wg,const bf*b,bf*y,int total,int n){
 int pair=blockIdx.x*blockDim.x+threadIdx.x;if(pair*2>=total)return;float2 v=sum_pair(p,splits,total,pair,(*ag)*(*wg));int c=pair*2%n;
 y[pair*2]=__float2bfloat16(v.x+(b?__bfloat162float(b[c]):0));y[pair*2+1]=__float2bfloat16(v.y+(b?__bfloat162float(b[c+1]):0));
}
__global__ void residual(const bf*x,bf*h,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)h[i]=__float2bfloat16(__bfloat162float(h[i])+__bfloat162float(x[i]));}
__global__ void gelu(const bf*x,float*y,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)y[i]=tanh_gelu(__bfloat162float(x[i]));}
__global__ void silu(const __half*g,const __half*u,float*y,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total){float v=__half2float(g[i]);y[i]=v/(1.f+expf(-v))*__half2float(u[i]);}}
__global__ void quant8(const float*x,__nv_fp8_e4m3*y,const float*s,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)y[i]=__nv_fp8_e4m3(fminf(fmaxf(x[i]/(*s),-448.f),448.f));}
// F7 preserves FP32 projection values until attention's original Q scaling/BF16 rounding.
__global__ void materialize(const float*p,int splits,const float*ag,const float*wg,const bf*b,float*y,int total,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total){float v=0;for(int s=0;s<splits;s++)v+=p[s*total+i];y[i]=v*((*ag)*(*wg))+(b?__bfloat162float(b[i%n]):0);}}
__global__ void castbf(const float*x,bf*y,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)y[i]=__float2bfloat16(x[i]);}
__global__ void gelu_quant8(const bf*x,__nv_fp8_e4m3*y,const float*scale,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total)y[i]=__nv_fp8_e4m3(fminf(fmaxf(tanh_gelu(__bfloat162float(x[i]))/(*scale),-448.f),448.f));}

// Generated exact FP16 normalization helper follows, without fused quantization.
// __NORM_HALF__

void run_split(int op,std::vector<int64_t>a,int rows,int n,double eps,int64_t stream){
 auto st=(cudaStream_t)stream;int total=rows*n;int blocks=(total+255)/256;
 #define P(T,I) reinterpret_cast<T*>(a[I])
 switch(op){
 case 0:norm_bf<<<rows,n/4,0,st>>>(P(bf,0),P(float,1),n,eps);break;
 case 1:mod<<<blocks,256,0,st>>>(P(float,0),P(bf,1),P(bf,2),total,n);break;
 case 2:quant4<false><<<(total/16+127)/128,128,0,st>>>(P(void,0),P(unsigned char,1),P(unsigned char,2),P(float,3),total);break;
 case 3:reduce_bias<<<(total/2+255)/256,256,0,st>>>(P(float,0),a[1],P(float,2),P(float,3),P(bf,4),P(bf,5),total,n);break;
 case 4:residual<<<blocks,256,0,st>>>(P(bf,0),P(bf,1),total);break;
 case 5:gelu<<<blocks,256,0,st>>>(P(bf,0),P(float,1),total);break;
 case 6:silu<<<blocks,256,0,st>>>(P(__half,0),P(__half,1),P(float,2),total);break;
 case 7:quant8<<<blocks,256,0,st>>>(P(float,0),P(__nv_fp8_e4m3,1),P(float,2),total);break;
 case 8:materialize<<<blocks,256,0,st>>>(P(float,0),a[1],P(float,2),P(float,3),P(bf,4),P(float,5),total,n);break;
 case 9:quant4<true><<<(total/16+127)/128,128,0,st>>>(P(void,0),P(unsigned char,1),P(unsigned char,2),P(float,3),total);break;
 case 10:half_control::norm_half<<<rows,n/8,0,st>>>(P(__half,0),nullptr,nullptr,nullptr,nullptr,P(__half,1),P(__half,2),a[3],eps,P(float,4),nullptr,P(__half,5),n);break;
 case 11:castbf<<<blocks,256,0,st>>>(P(float,0),P(bf,1),total);break;
 case 13:gelu_quant8<<<blocks,256,0,st>>>(P(bf,0),P(__nv_fp8_e4m3,1),P(float,2),total);break;
 }
 #undef P
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
