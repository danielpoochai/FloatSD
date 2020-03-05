import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fp8_cuda
from torch.utils.cpp_extension import load

__all__ = ['float_quantize', "quantizer", "Quantizer"]

######## FLOAT8 QUANT ######## 
######## exp_bits = 5 ######## 
######## man_bits = 2 ######## 
####### offset:(-24,7) ####### 

def assert_wl_fl(wl, fl, stage=""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))


def quantizer(forward_rounding="nearest", backward_rounding="nearest"):

    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)

    forward_quant = lambda x: fp8_cuda.float_quantize_nearest(x)
    backward_quant = lambda a: fp8_cuda.float_quantize_nearest(a)
    
    # if forward_rounding=="nearest":
    #     forward_quant = lambda x: fp8_cuda.float_quantize_nearest(x)
    # elif forward_rounding=="stochastic":
    #     forward_quant = lambda x: fp8_cuda.float_quantize_stochastic(x)
    

    # if backward_rounding=="nearest":
    #     backward_quant = lambda a: fp8_cuda.float_quantize_nearest(a)
    # elif backward_rounding=="stochastic":
    #     backward_quant = lambda a: fp8_cuda.float_quantize_stochastic(a)


    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            out = forward_quant(x.contiguous())
            return out

        @staticmethod
        def backward(self, grad_output):
            if self.needs_input_grad[0]:
                grad_input = backward_quant(grad_output.contiguous())
            else:
                grad_input = None

            return grad_input

    return Rounding.apply




def float_quantize(x, rounding="nearest"):
    
    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(rounding)

    # if rounding=="nearest":
    #     out = fp8_cuda.float_quantize_nearest(x.contiguous())
    # elif rounding=="stochastic":
    #     out = fp8_cuda.float_quantize_stochastic(x.contiguous())
        
    out = fp8_cuda.float_quantize_nearest(x.contiguous())  
    return out


class Quantizer(nn.Module):
    def __init__(self, forward_rounding="nearest", backward_rounding="nearest"):
        super(Quantizer, self).__init__()
        self.quantize = quantizer(forward_rounding, backward_rounding)

    def forward(self, x):
        return self.quantize(x)
