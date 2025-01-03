# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/Makefile
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This makefile does nothing but delegating the actual building to CMake.
CMAKE_BUILD_TYPE ?= Release
CMAKE_CXX_STANDARD ?= 20
FLATFLOW_BUILD_TESTS ?= OFF
FLATFLOW_ENABLE_ASAN ?= OFF
FLATFLOW_ENABLE_UBSAN ?= OFF

.PHONY: all generate test clean

all:
	@mkdir -p build && \
		cd build && \
		cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_CXX_STANDARD=$(CMAKE_CXX_STANDARD) \
		-DFLATFLOW_BUILD_TESTS=$(FLATFLOW_BUILD_TESTS) \
		-DFLATFLOW_ENABLE_ASAN=$(FLATFLOW_ENABLE_ASAN) \
		-DFLATFLOW_ENABLE_UBSAN=$(FLATFLOW_ENABLE_UBSAN) && \
		$(MAKE)

generate:
	@mkdir -p tmp && \
		mv flatflow/__init__.py tmp && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc -I . --keep-prefix flatflow/rpc/communicator.fbs && \
		./build/third_party/flatbuffers/flatc -p flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -p --grpc -I . flatflow/rpc/communicator.fbs && \
		./build/third_party/flatbuffers/flatc -c -o tests/data tests/data/dataset_test.fbs && \
		./build/third_party/flatbuffers/flatc -c -o tests/scheduler tests/scheduler/scheduler_test.fbs && \
		mv tmp/__init__.py flatflow && \
		mv flatflow/BroadcastRequest.py flatflow/rpc && \
		mv flatflow/BroadcastResponse.py flatflow/rpc && \
		mv flatflow/Empty.py flatflow/rpc && \
		mv flatflow/InitRequest.py flatflow/rpc && \
		mv flatflow/communicator_grpc_fb.py flatflow/rpc && \
		rmdir tmp

test:
	@ctest --test-dir build

clean:
	@rm -r build
