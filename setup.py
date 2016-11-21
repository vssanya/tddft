import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext = Extension("*", ["wrapper/*.pyx"],
                libraries=["tdse"],
                library_dirs=[
                    'build/src',
                ],
                include_dirs=[
                    'src',
                    numpy.get_include()
                ])
setup(
    name = "tdse",
    ext_modules = cythonize([ext])
)
