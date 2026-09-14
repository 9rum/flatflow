# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/Makefile
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This makefile does nothing but delegating the actual building to CMake.
CMAKE_BUILD_TYPE ?= Release
CMAKE_CXX_STANDARD ?= 20
FLATFLOW_BUILD_TESTS ?= ON
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
	@./build/third_party/flatbuffers/flatc -c -o flatflow/ops --scoped-enums --no-emit-min-max-enum-values flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops --scoped-enums --no-emit-min-max-enum-values flatflow/ops/scalar_type.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/scalar_type.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/ops -I . --scoped-enums --keep-prefix flatflow/ops/graph.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops -I . --gen-onefile --python-typing flatflow/ops/graph.fbs

.PHONY: degenerate
degenerate:
	@rm -f flatflow/ops/graph_generated.h \
		flatflow/ops/graph_generated.py \
		flatflow/ops/graph_generated.pyi \
		flatflow/ops/operator_generated.h \
		flatflow/ops/operator_generated.py \
		flatflow/ops/operator_generated.pyi \
		flatflow/ops/scalar_type_generated.h \
		flatflow/ops/scalar_type_generated.py \
		flatflow/ops/scalar_type_generated.pyi

.PHONY: check
check:
	@ctest --test-dir build

.PHONY: clean
clean:
	@rm -rf build
