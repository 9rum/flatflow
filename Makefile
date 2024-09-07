# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/Makefile
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This makefile does nothing but delegating the actual building to CMake.
.PHONY: all build generate test clean

all:
	@python3 -m build --wheel

build:
	@mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCMAKE_CXX_STANDARD=$(CMAKE_CXX_STANDARD) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CMAKE_LIBRARY_OUTPUT_DIRECTORY) -DFLATFLOW_BUILD_TESTS=$(FLATFLOW_BUILD_TESTS) && $(MAKE)

generate:
	@./build/third_party/flatbuffers/flatc -c -o flatflow/rpc flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc -I . --keep-prefix flatflow/rpc/communicator.fbs && \
		./build/third_party/flatbuffers/flatc -p flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -p --grpc -I . flatflow/rpc/communicator.fbs && \
		./build/third_party/flatbuffers/flatc -c -o tests/data tests/data/dataset_test.fbs && \
		./build/third_party/flatbuffers/flatc -c -o tests/scheduler tests/scheduler/scheduler_test.fbs

test:
	@ctest --test-dir build && \
		pytest

clean:
	@rm -r build
