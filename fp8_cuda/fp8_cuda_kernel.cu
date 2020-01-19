#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <torch/types.h>
#include <vector>
#include "bit_helper.cu"

__global__ void float_nearest_kernel(float* __restrict__ a, float* o, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
      unsigned int old_num = FLOAT_TO_BITS(&a[index]);
      unsigned int quantize = round_bitwise_nearest(old_num);
      quantize = clip_exponent(old_num, quantize);
      float quantize_float = BITS_TO_FLOAT(&quantize);
      o[index] = quantize_float;
    }
}


torch::Tensor float_quantize_nearest_cuda(torch::Tensor a){
    auto o = torch::zeros_like(a);
    int size = a.numel();
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;
  
    float_nearest_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                   o.data<float>(),
                                                   size);
    return o;
}

