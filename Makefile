# Adapted from https://github.com/pytorch/pytorch/blob/v2.2.0/Makefile
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This makefile does nothing but delegating the actual building to CMake.
.PHONY: all generate test clean

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

generate:
	@./build/third_party/flatbuffers/flatc -c -o flatflow/data/ flatflow/data/dataset_test.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/scheduler/ flatflow/scheduler/scheduler_test.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc/ flatflow/rpc/empty.fbs && \
		./build/third_party/flatbuffers/flatc -c -o flatflow/rpc/ -I . --grpc --keep-prefix flatflow/rpc/communicator.fbs

test:
	@ctest --test-dir build

clean:
	@rm -r build
