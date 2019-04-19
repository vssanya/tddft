import sys
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = '/usr/bin/mpicc'
os.environ['CXX'] = '/usr/bin/mpicxx'

import numpy
import mpi4py
from Cython.Build import cythonize
from Cython import Tempita as tempita

import hashlib

HASH_FILE = 'cythonize.dat'

build_requires = ['numpy', 'mpi4py', 'cython']

metadata = dict(
    name = "tdse",
    version = "0.1",
    author = "Romanov Alexander",
    author_email = "vssanya@yandex.ru",
    packages = build_requires
)

def load_hashes(filename):
    # Return { filename : (sha1 of input, sha1 of output) }
    if os.path.isfile(filename):
        hashes = {}
        with open(filename, 'r') as f:
            for line in f:
                filename, inhash, outhash = line.split()
                hashes[filename] = (inhash, outhash)
    else:
        hashes = {}
    return hashes

def save_hashes(hash_db, filename):
    with open(filename, 'w') as f:
        for key, value in sorted(hash_db.items()):
            f.write("%s %s %s\n" % (key, value[0], value[1]))

def sha1_of_file(filename):
    h = hashlib.sha1()
    with open(filename, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def normpath(path):
    path = path.replace(os.sep, '/')
    if path.startswith('./'):
        path = path[2:]
    return path

def get_hash(frompath, topath):
    from_hash = sha1_of_file(frompath)
    to_hash = sha1_of_file(topath) if os.path.exists(topath) else None
    return (from_hash, to_hash)

def process(path, fromfile, tofile, processor_function, hash_db):
    fullfrompath = os.path.join(path, fromfile)
    fulltopath = os.path.join(path, tofile)
    current_hash = get_hash(fullfrompath, fulltopath)
    if current_hash == hash_db.get(normpath(fullfrompath), None):
        print('%s has not changed' % fullfrompath)
        return

    orig_cwd = os.getcwd()
    try:
        os.chdir(path)
        print('Processing %s' % fullfrompath)
        processor_function(fromfile, tofile)
    finally:
        os.chdir(orig_cwd)
    # changed target file, recompute hash
    current_hash = get_hash(fullfrompath, fulltopath)
    # store hash in db
    hash_db[normpath(fullfrompath)] = current_hash

def process_tempita_pyx(fromfile, tofile):
    assert fromfile.endswith('.pyx.in')
    with open(fromfile, "r") as f:
        tmpl = f.read()
    pyxcontent = tempita.sub(tmpl)
    pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
    with open(pyxfile, "w") as f:
        f.write(pyxcontent)

def find_process_files(root_dir):
    hash_db = load_hashes(HASH_FILE)
    for cur_dir, dirs, files in os.walk(root_dir):
        for filename in files:
            in_file = os.path.join(cur_dir, filename + ".in")
            if filename.endswith('.pyx.in'):
                fromfile = filename
                tofile = filename[:-len('.in')]
                process(cur_dir, fromfile, tofile, process_tempita_pyx, hash_db)
                save_hashes(hash_db, HASH_FILE)

find_process_files("wrapper")

libraries=["tdse", "lapack"]
from ctypes.util import find_library
if find_library('cuda') is not None:
    print("Compile with CUDA")
    libraries.append['tdse_gpu']

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
                    '-std=gnu++17',
                    '-D_MPI',
                    '-fopenmp',
                    '-lmpi',
                    '-pthread',
                    '-g',
                ],
                language="c++",
                )

metadata['ext_modules'] = cythonize([ext], compiler_directives={'embedsignature': True})

setup(**metadata)
