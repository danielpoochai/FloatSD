# include <torch/extension.h>
#include <vector>


// cuda declarations
// a: exp   b: man-1    c:man-2
torch::Tensor floatSD_cuda_quantize(int expOffset, torch::Tensor a, torch::Tensor o);
// STU
std::vector<torch::Tensor> floatSD_cuda_STU( float lr, float momemtum, int expOffset,
                                       torch::Tensor velocity,
                                       torch::Tensor grad,
                                       torch::Tensor a);


// C++ interface
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor floatSD_quantize(int expOffset, torch::Tensor a, torch::Tensor o){
  CHECK_INPUT(a);
  CHECK_INPUT(o);
  return floatSD_cuda_quantize(expOffset, a, o);
}

std::vector<torch::Tensor> floatSD_STU(float lr, float momemtum, int expOffset,
                                       torch::Tensor velocity,
                                       torch::Tensor grad,
                                       torch::Tensor a){
  CHECK_INPUT(velocity);
  CHECK_INPUT(grad); 
  CHECK_INPUT(a);
  return floatSD_cuda_STU(lr, momemtum, expOffset, velocity, grad, a);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("floatSD_quant", &floatSD_quantize, "floatSD Quantization (CUDA)");
  m.def("STU", &floatSD_STU, "floatSD STU (CUDA)");
}
