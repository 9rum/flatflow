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

.PHONY: all
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

.PHONY: generate
generate:
	@mkdir -p tmp/ops && \
		mkdir -p tmp/rpc && \
		mv flatflow/ops/__init__.py tmp/ops && \
		mv flatflow/rpc/__init__.py tmp/rpc && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops --scoped-enums --no-emit-min-max-enum-values flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops --scoped-enums --no-emit-min-max-enum-values flatflow/ops/scalar_type.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/scalar_type.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops -I . --keep-prefix flatflow/ops/graph.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops -I . --gen-onefile --python-typing flatflow/ops/graph.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/rpc --gen-onefile --python-typing flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc -I . --keep-prefix --grpc flatflow/rpc/controlplane.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/rpc -I . --gen-onefile --python-typing --grpc --grpc-filename-suffix _fb flatflow/rpc/controlplane.fbs && \
		git apply flatflow/ops/graph_generated.diff && \
		rm flatflow/ops/Operator.py && \
		rm flatflow/ops/ScalarType.py && \
		rm flatflow/rpc/Operator.py && \
		rm flatflow/rpc/ScalarType.py && \
		mv tmp/ops/__init__.py flatflow/ops && \
		mv tmp/rpc/__init__.py flatflow/rpc && \
		rmdir tmp/ops && \
		rmdir tmp/rpc && \
		rmdir tmp

.PHONY: degenerate
degenerate:
	@rm flatflow/ops/graph_generated.h \
		flatflow/ops/graph_generated.py \
		flatflow/ops/graph_generated.pyi \
		flatflow/ops/operator_generated.h \
		flatflow/ops/operator_generated.py \
		flatflow/ops/operator_generated.pyi \
		flatflow/ops/scalar_type_generated.h \
		flatflow/ops/scalar_type_generated.py \
		flatflow/ops/scalar_type_generated.pyi \
		flatflow/rpc/controlplane.grpc.fb.cc \
		flatflow/rpc/controlplane.grpc.fb.h \
		flatflow/rpc/controlplane_generated.h \
		flatflow/rpc/controlplane_generated.py \
		flatflow/rpc/controlplane_generated.pyi \
		flatflow/rpc/controlplane_grpc_fb.py \
		flatflow/rpc/controlplane_grpc_fb.pyi \
		flatflow/rpc/empty_generated.h \
		flatflow/rpc/empty_generated.py \
		flatflow/rpc/empty_generated.pyi

.PHONY: check
check:
	@ctest --test-dir build

.PHONY: clean
clean:
	@rm -r build
