# For AArch64 build, use quay.io/pypa/manylinux_2_28_aarch64.
ARG BASE=quay.io/pypa/manylinux_2_28_x86_64

FROM ${BASE}

ENV PATH="/root/.cargo/bin:${PATH}"

# Install the Rust toolchain and maturin.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN pipx install maturin

# Download and install the FlatBuffers compiler.
# This should be consistent with the FlatBuffers version in requirements.txt.
RUN pushd $(mktemp -d) && \
    curl -LO https://github.com/google/flatbuffers/releases/download/v25.12.19/Linux.flatc.binary.g++-13.zip && \
    unzip Linux.flatc.binary.g++-13.zip && \
    install flatc /usr/local/bin/flatc && \
    popd

WORKDIR /workspace/flatflow

COPY . .

# Note: Even if the wheel is successfully built, `cargo test` may fail in manylinux.
# This is due to the absence of Python development package and can be resolved by
# installing the relevant package such as `yum install -y python3.12-devel`.
RUN make generate && \
    make build && \
    auditwheel repair target/wheels/*

# For PyPI upload, run the commands commented out below.
# RUN pipx install twine && \
#     twine upload wheelhouse/*
