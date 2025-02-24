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
	@mkdir -p tmp/ops && \
		mkdir -p tmp/rpc && \
		mv flatflow/ops/__init__.py tmp/ops && \
		mv flatflow/rpc/__init__.py tmp/rpc && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops --scoped-enums --no-emit-min-max-enum-values flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops -I . --keep-prefix flatflow/ops/node.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops -I . --gen-onefile --python-typing flatflow/ops/node.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops -I . --keep-prefix flatflow/ops/graph.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops -I . --gen-onefile --python-typing flatflow/ops/graph.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/rpc --gen-onefile --python-typing flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc -I . --grpc --keep-prefix flatflow/rpc/controlplane.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/rpc -I . --grpc --gen-onefile --python-typing --grpc-python-typed-handlers --grpc-filename-suffix _fb flatflow/rpc/controlplane.fbs && \
		mv tmp/ops/__init__.py flatflow/ops && \
		mv tmp/rpc/__init__.py flatflow/rpc && \
		rm flatflow/ops/Operator.py && \
		rm flatflow/rpc/Operator.py && \
		rmdir tmp/ops && \
		rmdir tmp/rpc && \
		rmdir tmp

test:
	@ctest --test-dir build

clean:
	@rm -r build
