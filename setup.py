import sys
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = 'mpicxx'
os.environ['CXX'] = 'mpicxx'

def parse_setuppy_commands():
    """Check the commands and respond appropriately.  Disable broken commands.
    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent("""
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup.py install`:
              - `pip install .`       (from a git repo or downloaded source
                                       release)
              - `pip install numpy`   (last NumPy release on PyPi)
            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            NumPy-specific help
            -------------------
            To install NumPy from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest NumPy release
            from PyPi, use `pip install numpy`.
            For help with build/installation issues, please ask on the
            numpy-discussion mailing list.  If you are sure that you have run
            into a bug, please report it at https://github.com/numpy/numpy/issues.
            Setuptools commands help
            ------------------------
            """))
        return False


    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test="""
            `setup.py test` is not supported.  Use one of the following
            instead:
              - `python runtests.py`              (to build and test)
              - `python runtests.py --no-build`   (to test installed numpy)
              - `>>> numpy.test()`           (run tests for installed numpy
                                              from within an interpreter)
            """,
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        upload_docs="`setup.py upload_docs` is not supported",
        easy_install="`setup.py easy_install` is not supported",
        clean="""
            `setup.py clean` is not supported, use one of the following instead:
              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        check="`setup.py check` is not supported",
        register="`setup.py register` is not supported",
        bdist_dumb="`setup.py bdist_dumb` is not supported",
        bdist="`setup.py bdist` is not supported",
        build_sphinx="""
            `setup.py build_sphinx` is not supported, use the
            Makefile under doc/""",
        flake8="`setup.py flake8` is not supported, use flake8 standalone",
        )
    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                     'register', 'check', 'install_data', 'install_headers',
                     'install_lib', 'install_scripts', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    import warnings
    warnings.warn("Unrecognized setuptools command, proceeding with "
                  "generating Cython sources and expanding templates", stacklevel=2)

    return True

build_requires = ['numpy', 'mpi4py', 'cython']

metadata = dict(
    name = "tdse",
    version = "0.1",
    author = "Romanov Alexander",
    author_email = "vssanya@yandex.ru",
    packages = build_requires
)

run_build = parse_setuppy_commands()

if run_build:
    import numpy
    import mpi4py
    from Cython.Build import cythonize

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
                        '-g',
                    ],
                    language="c++",
                    )

    metadata['ext_modules'] = cythonize([ext], compiler_directives={'embedsignature': True})

setup(**metadata)
