from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("graph_DollarGrad", ["graph_DollarGrad.pyx"], include_dirs=[".", get_include()], extra_compile_args=['-w'])
setup(name = "graph_DollarGrad",ext_modules=cythonize(ext))
