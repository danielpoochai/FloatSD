# include <torch/extension.h>
#include <vector>


// cuda declarations
torch::Tensor float_quantize_nearest_cuda(torch::Tensor a);
// torch::Tensor float_quantize_stochastic_cuda(torch::Tensor a);

// C++ interface
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor float_quantize_nearest(torch::Tensor a){
  CHECK_INPUT(a);
  return float_quantize_nearest_cuda(a);
}

// torch::Tensor float_quantize_stochastic(torch::Tensor a){
//   CHECK_INPUT(a);
//   return float_quantize_stochastic_cuda(a);
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("float_quantize_nearest", &float_quantize_nearest, "FP8 Quantization (CUDA)");
}
