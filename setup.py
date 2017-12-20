import numpy
import mpi4py

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os
os.environ['CC'] = 'mpicxx'
os.environ['CXX'] = 'mpicxx'

ext = Extension("*", ["wrapper/*.pyx"],
                libraries=["tdse", "lapack"],
                library_dirs=[
                    'build/src',
                ],
                include_dirs=[
                    'src',
                    numpy.get_include(),
                    mpi4py.get_include(),
                ],
                extra_compile_args=[
                    '-std=gnu++11',
                    '-D_MPI',
                    '-g',
                ],
                language="c++",
                )
setup(
    name = "tdse",
    ext_modules = cythonize([ext], compiler_directives={
        'embedsignature': True
        })
)
