.PHONY: all
all:
	@maturin develop

.PHONY: build
build:
	@maturin build --release

.PHONY: generate
generate:
	@flatc -r -o src/ops schema/operator.fbs && \
		flatc -p -o flatflow/ops --gen-onefile --python-typing schema/operator.fbs && \
		flatc -r -o src/ops schema/scalar_type.fbs && \
		flatc -p -o flatflow/ops --gen-onefile --python-typing schema/scalar_type.fbs && \
		flatc -r -o src/ops -I schema --include-prefix ops schema/graph.fbs && \
		flatc -p -o flatflow/ops -I schema --gen-onefile --python-typing schema/graph.fbs

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
