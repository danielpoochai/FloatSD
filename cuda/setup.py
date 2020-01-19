from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='floatSD_cuda',
    ext_modules=[
        CUDAExtension('floatSD_cuda', [
            'floatSD_cuda.cpp',
            'floatSD_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })