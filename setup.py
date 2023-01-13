#!/usr/bin/env python3

# This file is largely based on OpenQL (Apache 2.0 License, Copyright [2016] [Nader
# Khammassi & Imran Ashraf, QuTech, TU Delft]):
# https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
#
# For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/setup.py
#
# Changes were made by adding the get_version function, replacing openql (and the like)
# by qgym, and correctly configuring the arguments to setup. Furthermore, the code has
# been modernized (f-strings, pathlib, etc), since qgym only supports Python 3.8 and
# above. Lastly, the file was formatted following the black code style.

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
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.install import install as _install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

ROOT_DIR = Path.cwd()   # root of the repository
PYSRC_DIR = ROOT_DIR / "python"  # Python source files
TARGET_DIR = ROOT_DIR / "pybuild"  # python-specific build directory
BUILD_DIR = TARGET_DIR / "build"  # directory for setuptools to dump various files into
DIST_DIR = TARGET_DIR / "dist"  # wheel output directory
CBUILD_DIR = TARGET_DIR / "cbuild"  # cmake build directory
PREFIX_DIR = TARGET_DIR / "prefix"  # cmake install prefix
SRCMOD_DIR = PYSRC_DIR / "qgym"  # qgym Python module directory, source files only
MODULE_DIR = TARGET_DIR / "qgym"  # qgym Python module directory for editable install

# Copy the hand-written Python sources into the module directory that we're
# telling setuptools is our source directory, because setuptools insists on
# spamming output files into that directory. This is ugly, especially because
# it has to run before setup() is invoked, but seems to be more-or-less
# unavoidable to get editable installs to work.
if not TARGET_DIR.exists():
    TARGET_DIR.mkdir(parents=True)
copy_tree(str(SRCMOD_DIR), str(MODULE_DIR))


def read(fname):
    return (Path(__file__).parents[0] / fname).read_text()


# this method was added specifically for qgym to read the version from the __init__ file.
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


class clean(_clean):
    def run(self):
        _clean.run(self)
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)


class build_ext(_build_ext):
    def run(self):
        from plumbum import FG, ProcessExecutionError, local

        # If we were previously built in a different directory, nuke the cbuild
        # dir to prevent inane CMake errors. This happens when the user does
        # pip install . after building locally.
        if (CBUILD_DIR / "CMakeCache.txt").exists():
            for line in (CBUILD_DIR / "CMakeCache.txt").read_text().split("\n"):
                line = line.split("#")[0].strip()
                if line and line.startswith("QGym_BINARY_DIR:STATIC"):
                    config_dir = line.split("=", maxsplit=1)[1]
                    if os.path.realpath(config_dir) != os.path.realpath(CBUILD_DIR):
                        print("removing pybuild/cbuild to avoid CMakeCache error")
                        shutil.rmtree(CBUILD_DIR)
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
        if not CBUILD_DIR.exists():
            os.makedirs(CBUILD_DIR)
        with local.cwd(str(CBUILD_DIR)):
            build_type = os.environ.get("QGYM_BUILD_TYPE", "Release")

            cmd = (
                local["cmake"][str(ROOT_DIR)]["-DQGYM_BUILD_PYTHON=YES"][
                    f"-DCMAKE_INSTALL_PREFIX={PREFIX_DIR}"
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
                    local["cmake"][str(PYSRC_DIR / "compat" / "test-cmake-config")] & FG
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
                log.info(f" - Python is {64 if python_is_64 else 32}-bit")
                log.info(f" - CMake is building {64 if cmake_is_64 else 32}-bit by default")
                log.info(f" - CMake {'IS' if cmake_is_msvc else 'is NOT'} building using MSVC")
                log.info(f" - MSVC {'IS' if msvc_is_fixed_to_64 else 'is NOT'} fixed to 64-bit")

                # If there's a mismatch, see what we can do.
                if python_is_64 != cmake_is_64:
                    if msvc_is_fixed_to_64 and not python_is_64:
                        raise RuntimeError(
                            "MSVC is configured to build 64-bit binaries, but Python is 32-bit!"
                        )
                    if not cmake_is_msvc:
                        raise RuntimeError(
                            "Mismatch in 32-bit/64-bit between CMake defaults "
                            f"({64 if cmake_is_64 else 32}-bit) and Python install "
                            f"({64 if python_is_64 else 32}-bit)!"
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
        self.build_base = os.path.relpath(BUILD_DIR)

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
        self.dist_dir = os.path.relpath(DIST_DIR)


class bdist_wheel(_bdist_wheel):
    def run(self):
        if platform.system() == "Darwin":
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.10"
        _bdist_wheel.run(self)
        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
        wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        if platform.system() == "Darwin":
            from delocate.delocating import delocate_wheel

            delocate_wheel(wheel_path)


class sdist(_sdist):
    def finalize_options(self):
        _sdist.finalize_options(self)
        self.dist_dir = os.path.relpath(DIST_DIR)


class egg_info(_egg_info):
    def initialize_options(self):
        _egg_info.initialize_options(self)
        self.egg_base = os.path.relpath(TARGET_DIR)


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
        "qgym.templates",
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
