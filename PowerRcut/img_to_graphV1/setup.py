from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("img_to_graph", ["img_to_graph.pyx"], include_dirs=[".", get_include()], extra_compile_args=['-w'])
setup(name = "img_to_graph",ext_modules=cythonize(ext))
