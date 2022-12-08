#!/usr/bin/env python3

# This file is largely based on OpenQL (Apache 2.0 License, Copyright [2016] [Nader Khammassi & Imran Ashraf, QuTech, TU Delft]): https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
# For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/setup.py
# Changes were made by adding the get_version function, replacing openql (and the like) by qgym, and correctly
# configuring the arguments to setup. Also the file was formatted following the black code style.

import os
import platform
import shutil
import sys
from distutils import log
from distutils.command.bdist import bdist as _bdist
from distutils.command.build import build as _build
from distutils.command.clean import clean as _clean
from distutils.command.sdist import sdist as _sdist
from distutils.dir_util import copy_tree

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.install import install as _install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

root_dir = os.getcwd()  # root of the repository
src_dir = root_dir + os.sep + "src"  # C++ source directory
inc_dir = root_dir + os.sep + "include"  # C++ include directory
pysrc_dir = root_dir + os.sep + "python"  # Python source files
target_dir = root_dir + os.sep + "pybuild"  # python-specific build directory
build_dir = (
    target_dir + os.sep + "build"
)  # directory for setuptools to dump various files into
dist_dir = target_dir + os.sep + "dist"  # wheel output directory
cbuild_dir = target_dir + os.sep + "cbuild"  # cmake build directory
prefix_dir = target_dir + os.sep + "prefix"  # cmake install prefix
srcmod_dir = (
    pysrc_dir + os.sep + "qgym"
)  # qgym Python module directory, source files only
module_dir = (
    target_dir + os.sep + "qgym"
)  # qgym Python module directory for editable install

# Copy the hand-written Python sources into the module directory that we're
# telling setuptools is our source directory, because setuptools insists on
# spamming output files into that directory. This is ugly, especially because
# it has to run before setup() is invoked, but seems to be more-or-less
# unavoidable to get editable installs to work.
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
copy_tree(srcmod_dir, module_dir)


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


# this method was added specifically for qgym to read the version from the __init__ file.
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


class clean(_clean):
    def run(self):
        _clean.run(self)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)


class build_ext(_build_ext):
    def run(self):
        from plumbum import FG, ProcessExecutionError, local

        # If we were previously built in a different directory, nuke the cbuild
        # dir to prevent inane CMake errors. This happens when the user does
        # pip install . after building locally.
        if os.path.exists(cbuild_dir + os.sep + "CMakeCache.txt"):
            with open(cbuild_dir + os.sep + "CMakeCache.txt", "r") as f:
                for line in f.read().split("\n"):
                    line = line.split("#")[0].strip()
                    if not line:
                        continue
                    if line.startswith("QGym_BINARY_DIR:STATIC"):
                        config_dir = line.split("=", maxsplit=1)[1]
                        if os.path.realpath(config_dir) != os.path.realpath(cbuild_dir):
                            print("removing pybuild/cbuild to avoid CMakeCache error")
                            shutil.rmtree(cbuild_dir)
                        break

        # Figure out how many parallel processes to build with.
        if self.parallel:
            nprocs = str(self.parallel)
        else:
            nprocs = os.environ.get("NPROCS", "1")

        # Figure out how setuptools wants to name the extension file and where
        # it wants to place it.
        target = os.path.abspath(self.get_ext_fullpath("qgym._qgym"))

        # Build the Python extension and "install" it where setuptools expects
        # it.
        if not os.path.exists(cbuild_dir):
            os.makedirs(cbuild_dir)
        with local.cwd(cbuild_dir):
            build_type = os.environ.get("QGYM_BUILD_TYPE", "Release")

            cmd = (
                local["cmake"][root_dir]["-DQGYM_BUILD_PYTHON=YES"][
                    "-DCMAKE_INSTALL_PREFIX=" + prefix_dir
                ]["-DQGYM_PYTHON_DIR=" + os.path.dirname(target)][
                    "-DQGYM_PYTHON_EXT=" + os.path.basename(target)
                ]
                # Make sure CMake uses the Python installation corresponding
                # with the the Python version we're building with now.
                ["-DPYTHON_EXECUTABLE=" + sys.executable]
                # (ab)use static libs for the intermediate libraries to avoid
                # dealing with R(UN)PATH nonsense on Linux/OSX as much as
                # possible.
                ["-DBUILD_SHARED_LIBS=NO"]
                # Build type can be set using an environment variable.
                ["-DCMAKE_BUILD_TYPE=" + build_type]
            )

            # If we're on Windows, we're probably building with MSVC. In that
            # case, we might have to tell CMake whether we want to build for
            # x86 or x64, but depending on how MSVC is configured, that same
            # command-line option could also return an error. So we need to be
            # careful here.
            if platform.system() == "Windows":
                log.info("Trying to figure out bitness...")

                # Figure out what CMake is doing by default.
                if not os.path.exists("test-cmake-config"):
                    os.makedirs("test-cmake-config")
                with local.cwd("test-cmake-config"):
                    (
                        local["cmake"][
                            pysrc_dir + os.sep + "compat" + os.sep + "test-cmake-config"
                        ]
                        & FG
                    )
                    with open("values.cfg", "r") as f:
                        void_ptr_size, generator, *_ = f.read().split("\n")
                        cmake_is_64 = int(void_ptr_size.strip()) == 8
                        cmake_is_msvc = "Visual Studio" in generator
                        msvc_is_fixed_to_64 = cmake_is_msvc and (
                            "Win64" in generator or "IA64" in generator
                        )

                # Figure out what Python needs.
                python_is_64 = sys.maxsize > 2**32

                log.info("Figured out the following things:")
                log.info(" - Python is {}-bit".format(64 if python_is_64 else 32))
                log.info(
                    " - CMake is building {}-bit by default".format(
                        64 if cmake_is_64 else 32
                    )
                )
                log.info(
                    " - CMake {} building using MSVC".format(
                        "IS" if cmake_is_msvc else "is NOT"
                    )
                )
                log.info(
                    " - MSVC {} fixed to 64-bit".format(
                        "IS" if msvc_is_fixed_to_64 else "is NOT"
                    )
                )

                # If there's a mismatch, see what we can do.
                if python_is_64 != cmake_is_64:
                    if msvc_is_fixed_to_64 and not python_is_64:
                        raise RuntimeError(
                            "MSVC is configured to build 64-bit binaries, but Python is 32-bit!"
                        )
                    if not cmake_is_msvc:
                        raise RuntimeError(
                            "Mismatch in 32-bit/64-bit between CMake defaults ({}) and Python install ({})!".format(
                                "64-bit" if cmake_is_64 else "32-bit",
                                "64-bit" if python_is_64 else "32-bit",
                            )
                        )

                    # Looks like we're compiling with MSVC, and MSVC is merely
                    # defaulting to the wrong bitness, which means we should be
                    # able to change it with the -A flag.
                    if python_is_64:
                        cmd = cmd["-A"]["x64"]
                    else:
                        cmd = cmd["-A"]["win32"]

            # Run cmake configuration.
            cmd & FG

            # Do the build with the given number of parallel threads.
            build_cmd = local["cmake"]["--build"]["."]["--config"][build_type]
            cmd = build_cmd
            if nprocs != "1":
                try:
                    parallel_supported = tuple(
                        local["cmake"]("--version")
                        .split("\n")[0]
                        .split()[-1]
                        .split(".")
                    ) >= (3, 12)
                except:
                    parallel_supported = False
                if parallel_supported:
                    cmd = cmd["--parallel"][nprocs]
                elif not sys.platform.startswith("win"):
                    cmd = cmd["--"]["-j"][nprocs]
            cmd & FG

            # Do the install.
            try:
                # install target for makefiles
                build_cmd["--target"]["install"] & FG
            except ProcessExecutionError:
                # install target for MSVC
                build_cmd["--target"]["INSTALL"] & FG


class build(_build):
    def initialize_options(self):
        _build.initialize_options(self)
        self.build_base = os.path.relpath(build_dir)

    def run(self):
        # Make sure the extension is built before the Python module is "built",
        # otherwise SWIG's generated module isn't included.
        # See https://stackoverflow.com/questions/12491328
        self.run_command("build_ext")
        _build.run(self)


class install(_install):
    def run(self):
        # See https://stackoverflow.com/questions/12491328
        self.run_command("build_ext")
        _install.run(self)


class bdist(_bdist):
    def finalize_options(self):
        _bdist.finalize_options(self)
        self.dist_dir = os.path.relpath(dist_dir)


class bdist_wheel(_bdist_wheel):
    def run(self):
        if platform.system() == "Darwin":
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.10"
        _bdist_wheel.run(self)
        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = "{}-{}-{}-{}".format(
            self.wheel_dist_name, impl_tag, abi_tag, plat_tag
        )
        wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        if platform.system() == "Darwin":
            from delocate.delocating import delocate_wheel

            delocate_wheel(wheel_path)


class sdist(_sdist):
    def finalize_options(self):
        _sdist.finalize_options(self)
        self.dist_dir = os.path.relpath(dist_dir)


class egg_info(_egg_info):
    def initialize_options(self):
        _egg_info.initialize_options(self)
        self.egg_base = os.path.relpath(target_dir)


setup(
    name="qgym",
    version=get_version("python/qgym/__init__.py"),
    description="Reinforcement Learning Gym for OpenQL",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="QuTech (TNO, TU Delft)",
    maintainer="QuTech (TNO, TU Delft)",
    license="Apache License, Version 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Private :: Do Not Upload to pypi server",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 ",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience:: Developers",
        "Intended Audience:: Information Technology",
        "Intended Audience:: Science / Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "Reinforcement Learning",
        "QuTech",
        "TNO",
        "TU Delft",
        "Quantum",
        "Gym",
        "Quantum Compilation",
    ],
    packages=[
        "qgym",
        "qgym.envs",
        "qgym.envs.initial_mapping",
        "qgym.envs.scheduling",
        "qgym.spaces",
        "qgym.utils",
    ],
    package_dir={"": "pybuild"},
    # NOTE: the library build process is completely overridden to let CMake
    # handle it; setuptools' implementation is horribly broken. This is here
    # just to have the rest of setuptools understand that this is a Python
    # module with an extension in it.
    ext_modules=[Extension("qgym._qgym", [])],
    cmdclass={
        "bdist": bdist,
        "bdist_wheel": bdist_wheel,
        "build_ext": build_ext,
        "build": build,
        "install": install,
        "clean": clean,
        "egg_info": egg_info,
        "sdist": sdist,
    },
    python_requires=">=3.8,<3.10",
    setup_requires=[
        "plumbum",
        'delocate; platform_system == "Darwin"',
    ],
    install_requires=[
        'msvc-runtime; platform_system == "Windows"',
        "gym~=0.19.0",
        "networkx[default]~=2.7.1",
        "numpy~=1.22.3",
        "pygame~=2.1.2",
        "qutechopenql==0.10.0",  # keep this in sync with the submodule
        "scipy~=1.8.0",
    ],
    tests_require=[
        "pytest",
        "stable-baselines3~=1.4.0",
    ],
    extras_require={
        "tutorial": [
            "matplotlib~=3.5.3",
            "notebook",
            "stable_baselines3~=1.4.0",
        ],
        "dev": [
            "black",
            "isort",
            "matplotlib~=3.5.3",
            "notebook",
            "pytest",
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-math-dollar",
            "sphinx-rtd-theme~=1.0",
            "stable-baselines3~=1.4.0",
        ],
    },
    zip_safe=False,
)
