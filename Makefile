# Adapted from https://github.com/pytorch/pytorch/blob/v2.2.0/Makefile
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This makefile does nothing but delegating the actual building to CMake.
.PHONY: all test clean

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

test:
	@ctest --test-dir build

clean:
	@rm -r build
