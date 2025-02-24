# Copyright 2024 The FlatFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from typing import Optional

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(
        self,
        name: str,
        cmake_lists_dir: str = os.curdir,
        cmake_build_type: str = "Release",
        cmake_generator: str = "Ninja",
        cmake_args: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        self.cmake_build_type = cmake_build_type
        self.cmake_generator = cmake_generator
        if cmake_args is None:
            cmake_args = []
        self.cmake_args = cmake_args


class BuildExtension(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp, exist_ok=True)

        outdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        args = [
            "cmake",
            ext.cmake_lists_dir,
            "-G",
            ext.cmake_generator,
            f"-DCMAKE_BUILD_TYPE={ext.cmake_build_type}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={self.build_temp}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            *ext.cmake_args,
        ]
        cmd = " ".join(args)
        logging.info(f"-- Configuring: {cmd}")
        subprocess.check_call(args, cwd=self.build_temp)

        args = [
            "cmake",
            "--build",
            os.curdir,
            "--config",
            ext.cmake_build_type,
            "-j",
        ]
        cmd = " ".join(args)
        logging.info(f"-- Building: {cmd}")
        subprocess.check_call(args, cwd=self.build_temp)


cwd = os.path.dirname(__file__)


def get_flatflow_version() -> str:
    filename = os.path.join(cwd, "flatflow", "__init__.py")
    tree = ast.parse(open(filename).read(), filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    return str(node.value.s)
    raise RuntimeError("Unable to find version")


readme = open(os.path.join(cwd, "README.md")).read().strip()
requirements = open(os.path.join(cwd, "requirements.txt")).read().strip().split("\n")

ext_modules = [
    CMakeExtension(
        "flatflow._C",
        cmake_args=[
            "-DCMAKE_CXX_STANDARD=20",
            "-DFLATFLOW_BUILD_TESTS=ON",
            "-DFLATFLOW_ENABLE_ASAN=OFF",
            "-DFLATFLOW_ENABLE_UBSAN=OFF",
        ],
        py_limited_api=True,
    ),
]

cmdclass = {
    "build_ext": BuildExtension,
}

setup(
    name="flatflow",
    version=get_flatflow_version(),
    description="A learned system for parallel training of neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="The FlatFlow Authors",
    url="https://github.com/9rum/flatflow",
    packages=find_packages(exclude=("tests*",)),
    ext_modules=ext_modules,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache-2.0",
    cmdclass=cmdclass,
    install_requires=requirements,
    python_requires=">=3.9",
)
