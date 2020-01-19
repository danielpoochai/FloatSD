#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <torch/types.h>
#include <vector>

/////////////////////
// FLOATSD FUNCTION//
/////////////////////

__global__ void floatSD_kernel(int expOffset, int* __restrict__ a, float* o, int size) {
    // a: exp; b: group1; c: group2 

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // map index to 1 if init is 0
        float dict1[8] = {-pow(2,-1), -pow(2,-1), -pow(2,-2), -pow(2,-3), 0, pow(2,-3), pow(2,-2), pow(2,-1)};
        float dict2[8] = {-pow(2,-4), -pow(2,-4), -pow(2,-5), -pow(2,-6), 0, pow(2,-6), pow(2,-5), pow(2,-4)};
        float group_1 = dict1[int( a[index] >> 21 ) % 8];
        float group_2 = dict2[int( a[index] >> 18 ) % 8];
        int exponent = expOffset + (int( a[index] >> 24 ) % 4);
        
        o[index] = (group_1 + group_2) * pow(2,exponent);

    }
}

torch::Tensor floatSD_cuda_quantize(int expOffset, torch::Tensor a, torch::Tensor o){
    int size = a.numel();
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;
  
    floatSD_kernel<<<blockNums, blockSize>>>(expOffset, 
                                             a.data<int>(),
                                             o.data<float>(),
                                             size);
    return o;
  }


__global__ void STU_kernel(float lr, float momemtum,
                           int expOffset,
                           float* __restrict__ velocity,
                           float* __restrict__ grad,
                           int* __restrict__ a,  
                           int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // implement STU here
        int maxIdx = 7;
        int minIdx = 1;
        int minExp = 0;
        int maxExp = 3;
        int new_mastercopy[9];
        new_mastercopy[0] = int( a[index] >> 24 ) % 4;
        new_mastercopy[1] = int( a[index] >> 21 ) % 8;
        new_mastercopy[2] = int( a[index] >> 18 ) % 8;
        new_mastercopy[3] = int( a[index] >> 15 ) % 8;
        new_mastercopy[4] = int( a[index] >> 12 ) % 8;
        new_mastercopy[5] = int( a[index] >> 9 ) % 8;
        new_mastercopy[6] = int( a[index] >> 6 ) % 8;
        new_mastercopy[7] = int( a[index] >> 3 ) % 8;
        new_mastercopy[8] = int(a[index]) % 8;

        for (int i=1; i<=8; i++){
            if (new_mastercopy[i] == 0)
                new_mastercopy[i] = 1;
        }
        
        float delta = (grad[index]*lr-momemtum*velocity[index])*pow(2,-(new_mastercopy[0]+expOffset));
        velocity[index] = (momemtum*velocity[index] - grad[index]*lr);


        int s = 0; // carry or borrow
        if (delta < 0)
            s = 1;
        else s = -1;
        delta = abs(delta);

        // find MSG
        int k;
        if (delta >= pow(2,-3))
            k = 1;
        else if (delta >= pow(2,-6))
            k = 2;
        else if (delta >= pow(2,-9))
            k = 3;
        else if (delta >= pow(2,-12))
            k = 4;
        else if (delta >= pow(2,-15))
            k = 5;
        else if (delta >= pow(2,-18))
            k = 6;
        else if (delta >= pow(2,-21))
            k = 7;
        else if (delta >= pow(2,-24))
            k = 8;
	else return;

        // triger iteration
        bool stop = false;

        while(stop == false){
            if (k > 1){ // not MSG
                if ((s == 1) && (new_mastercopy[k] == maxIdx)){
                    new_mastercopy[k] = minIdx;
                    k = k-1; 
                }
                else if ((s == -1) && (new_mastercopy[k] == minIdx)){
                    new_mastercopy[k] = maxIdx;
                    k = k-1;
                }
                else{
                    new_mastercopy[k] = new_mastercopy[k] + s; 
                    stop = true; 
                }
            }
            else{ // MSG
                if (((s == 1) && (new_mastercopy[k] == maxIdx)) || ((s == -1) && (new_mastercopy[k] == minIdx))){
                    if (new_mastercopy[0] == maxExp) 
                        stop = true;
                    else{
                        for (int i=0; i<8; i++){ // non-MSG
                            if (2*new_mastercopy[k] > (maxIdx+1))
                                new_mastercopy[k] = new_mastercopy[k]-1;
                            else if (2*new_mastercopy[k] < (maxIdx+1))
                                new_mastercopy[k] = new_mastercopy[k]+1;
                        }
                        new_mastercopy[0] = new_mastercopy[0]+1;
                        stop = true;
                    }
                }
                else{
                    new_mastercopy[1] = new_mastercopy[1]+s;
                    stop = true;
                }
            }
        }   
        
        a[index] = (new_mastercopy[0]<<24) + (new_mastercopy[1]<<21) + (new_mastercopy[2]<<18)
                    + (new_mastercopy[3]<<15) + (new_mastercopy[4]<<12) + (new_mastercopy[5]<<9) 
                    + (new_mastercopy[6]<<6) + (new_mastercopy[7]<<3) + (new_mastercopy[8]);

    }
}

std::vector<torch::Tensor> floatSD_cuda_STU(  float lr, float momemtum, int expOffset,
                                                torch::Tensor velocity,
                                                torch::Tensor grad,
                                                torch::Tensor a)
{
    int size = a.numel();
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;
  
    STU_kernel<<<blockNums, blockSize>>>(    lr, momemtum, expOffset,
                                             velocity.data<float>(),
                                             grad.data<float>(),
                                             a.data<int>(),
                                             size);
    return {a, velocity};
}
