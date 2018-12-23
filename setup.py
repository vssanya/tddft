import sys
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = 'mpicxx'
os.environ['CXX'] = 'mpicxx'

import numpy
import mpi4py
from Cython.Build import cythonize
from Cython import Tempita as tempita

build_requires = ['numpy', 'mpi4py', 'cython']

metadata = dict(
    name = "tdse",
    version = "0.1",
    author = "Romanov Alexander",
    author_email = "vssanya@yandex.ru",
    packages = build_requires
)

def process_tempita_pyx(fromfile, cwd):
    from_filename = tempita.Template.from_filename
    template = from_filename(os.path.join(cwd, fromfile),
                             encoding=sys.getdefaultencoding())
    pyxcontent = template.substitute()
    assert fromfile.endswith('.pyx.in')
    pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
    with open(os.path.join(cwd, pyxfile), "w") as f:
        f.write(pyxcontent)

def find_process_files():
    for cur_dir, dirs, files in os.walk("wrapper"):
        for filename in files:
            if filename.endswith('.pyx.in'):
                process_tempita_pyx(filename, cur_dir)

find_process_files()

ext = Extension("*", ["wrapper/*.pyx"],
                libraries=["tdse", "lapack", "tdse_gpu"],
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
                    '-fopenmp',
                    '-g',
                ],
                language="c++",
                )

metadata['ext_modules'] = cythonize([ext], compiler_directives={'embedsignature': True})

setup(**metadata)
