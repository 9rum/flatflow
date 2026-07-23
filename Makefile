# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/Makefile
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This makefile does nothing but delegating the actual building to CMake.
CMAKE_BUILD_TYPE ?= Release
CMAKE_CXX_STANDARD ?= 20

.PHONY: all
all:
	@mkdir -p build && \
		cd build && \
		cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_CXX_STANDARD=$(CMAKE_CXX_STANDARD) && \
		$(MAKE)

.PHONY: generate
generate:
	@./build/third_party/flatbuffers/flatc -r -o src/ops flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/operator.fbs && \
		./build/third_party/flatbuffers/flatc -r -o src/ops flatflow/ops/scalar_type.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops --gen-onefile --python-typing flatflow/ops/scalar_type.fbs && \
		./build/third_party/flatbuffers/flatc -r -o src/ops -I . --include-prefix ops flatflow/ops/graph.fbs && \
		./build/third_party/flatbuffers/flatc -p -o flatflow/ops -I . --gen-onefile --python-typing flatflow/ops/graph.fbs

.PHONY: degenerate
degenerate:
	@rm flatflow/ops/graph_generated.py \
		flatflow/ops/graph_generated.pyi \
		flatflow/ops/operator_generated.py \
		flatflow/ops/operator_generated.pyi \
		flatflow/ops/scalar_type_generated.py \
		flatflow/ops/scalar_type_generated.pyi \
		src/ops/graph_generated.rs \
		src/ops/operator_generated.rs \
		src/ops/scalar_type_generated.rs

.PHONY: check
check:
	@ctest --test-dir build

.PHONY: clean
clean:
	@rm -r build
