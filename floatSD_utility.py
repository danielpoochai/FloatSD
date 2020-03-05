import torch
import numpy as np
import random
import math
import floatSD_cuda
            
       
torch.cuda.manual_seed(1234)  
def  randGroup(shape):
    g1 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**21 
    g2 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**18
    g3 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**15
    g4 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**12
    g5 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**9
    g6 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**6
    g7 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()*2**3
    g8 = torch.randint(1, 7, shape, dtype = torch.int32).cuda()
    copy = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8
    return copy

class floatSD():
    def __init__(self, weight, exp_off):
        
	    # random mantissa + 0 exponent 
        self.master_copy = randGroup(weight.shape)

        self.fp32_weight = torch.zeros(weight.shape, dtype = torch.float32).cuda() 

        # convert floatSD(3,3,2) to FP32
        self.fp32_weight = floatSD_cuda.floatSD_quant(exp_off, self.master_copy, self.fp32_weight)
        # print(self.fp32_weight)


