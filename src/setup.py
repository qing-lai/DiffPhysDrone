from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quadsim_cuda', #安装项目名
    ext_modules=[
        CUDAExtension('quadsim_cuda', [ #实际用于import的扩展模块名
            'quadsim.cpp',
            'quadsim_kernel.cu',
            'dynamics_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
