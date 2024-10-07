from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_extension',
    ext_modules=[
        CppExtension('my_extension', ['my_extension.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
