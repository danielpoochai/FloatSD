from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fp8_cuda',
    ext_modules=[
        CUDAExtension('fp8_cuda', [
            'fp8_cuda.cpp',
            'fp8_cuda_kernel.cu',
            'bit_helper.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })