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

import os
import subprocess
import sys

from setuptools import find_packages, setup

cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cwd)

from flatflow import __version__  # noqa: E402

readme = open(os.path.join(cwd, "README.md")).read().strip()
requirements = open(os.path.join(cwd, "requirements.txt")).read().strip().split("\n")

CMAKE_BUILD_TYPE = "Release"
CMAKE_CXX_STANDARD = 20
CMAKE_LIBRARY_OUTPUT_DIRECTORY = os.path.join(cwd, "flatflow")
FLATFLOW_BUILD_TESTS = "OFF"

subprocess.check_call(
    [
        "make",
        "build",
        f"CMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
        f"CMAKE_CXX_STANDARD={CMAKE_CXX_STANDARD}",
        f"CMAKE_LIBRARY_OUTPUT_DIRECTORY={CMAKE_LIBRARY_OUTPUT_DIRECTORY}",
        f"FLATFLOW_BUILD_TESTS={FLATFLOW_BUILD_TESTS}",
    ],
    cwd=cwd,
)

package_data = {"flatflow": ["*.so"]}

setup(
    name="flatflow",
    version=__version__,
    description="A learned system for parallel training of neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="The FlatFlow Authors",
    url="https://github.com/9rum/flatflow",
    packages=find_packages(exclude=("tests*",)),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache-2.0",
    package_data=package_data,
    install_requires=requirements,
    python_requires=">=3.8",
)
