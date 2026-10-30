import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
SOURCE=Path('/data/inference-baseline/third_party/FlashRT')
BUILD=HERE.parent/'reports/build'

def extension():
 BUILD.mkdir(parents=True,exist_ok=True)
 s=(SOURCE/'csrc/kernels/fp8_prefill_fp16.cu').read_text()
 a=s.index('__device__ __forceinline__ float warp_sum');b=s.index('__global__ void __launch_bounds__(256)\nbias_gelu_quant_kernel')
 frag=s[a:b].replace('residual_norm_quant_kernel','norm_half').replace('__nv_fp8_e4m3* __restrict__ out_fp8','float* __restrict__ out_fp8')
 frag=frag.replace('*reinterpret_cast<uint2*>(out_fp8 + off) = e4m3x8(v, 1.0f / fmaxf(*scale, 1e-12f));','for(int k=0;k<8;k++) out_fp8[off+k]=v[k];')
 code=(HERE/'split.cu').read_text().replace('// __NORM_HALF__','namespace half_control {\n'+frag+'\n}')
 (BUILD/'split_generated.cu').write_text('#include <stdexcept>\n'+code)
 bind='''#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
void run_split(int,std::vector<int64_t>,int,int,double,int64_t);
at::Tensor view(int64_t p,std::vector<int64_t>s,int dtype){return at::from_blob((void*)p,s,at::TensorOptions().device(at::kCUDA).dtype((at::ScalarType)dtype));}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run_split);m.def("view",&view);}
'''
 (BUILD/'bind.cpp').write_text(bind)
 os.environ['MAX_JOBS']='2';os.environ['TORCH_CUDA_ARCH_LIST']='12.0a'
 return load(name='flashrt_ablation_split',sources=[str(BUILD/'bind.cpp'),str(BUILD/'split_generated.cu')],build_directory=str(BUILD),extra_include_paths=[str(SOURCE/'csrc/gemm/fp4')],extra_cuda_cflags=['-O3','-lineinfo','--use_fast_math'],verbose=False)
