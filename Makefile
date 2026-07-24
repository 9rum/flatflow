.PHONY: all
all:
	@maturin develop

.PHONY: build
build:
	@maturin build --release

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
	@cargo test -- --show-output

.PHONY: clean
clean:
	@cargo clean
