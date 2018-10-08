# run the following command
# pip install --user .

from distutils.core import setup, Extension, DEBUG

sfc_module = Extension('nnet', sources = ['source.cpp', 'nnet.cpp', 'gpu.cpp'])

setup(name = 'nnet', version = '1.0',
    description = 'Python Package with nnet C++ extension',
    ext_modules = [sfc_module]
    )