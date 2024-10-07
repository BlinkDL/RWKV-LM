from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='mm_fp16i8',
    ext_modules=[
        CppExtension(
            'mm_fp16i8',
            ['mm_fp16i8.cpp'],
            extra_compile_args=[
                '-O3',
                '-mcpu=cortex-a76',
                '-march=armv8.2-a+fp16',
                '-std=c++17',
                '-fopenmp'    # Enable OpenMP
            ],
            extra_link_args=['-fopenmp'],  # Link against OpenMP library
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
