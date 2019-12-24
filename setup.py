from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

ext_modules=[
        CUDAExtension('dwconv._C',
            [
                'csrc/vision.cpp',
                'csrc/cuda/DepthWiseConv2d_cuda.cu',
                ],
            extra_compile_args=['-DWITH_CUDA', '-O2'],
            )
        ]

setup(
        name='dwconv',
        version='0.0.1',
        description='dwconv',
        author='Samson',
        author_email='samson.c.wang@gmail.com',
#        url='https://github.com/iHateTa11B0y/CROIUtils',
        packages=['dwconv'],
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
)
