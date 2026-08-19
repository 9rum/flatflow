# For AArch64 build, use quay.io/pypa/manylinux_2_28_aarch64.
ARG BASE=quay.io/pypa/manylinux_2_28_x86_64

FROM ${BASE}

# Install the Rust toolchain and maturin.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    source $HOME/.cargo/env
RUN pipx install maturin

# Download and install the FlatBuffers compiler.
# This should be consistent with the FlatBuffers version in pyproject.toml.
RUN pushd $(mktemp -d) && \
    curl -LO https://github.com/google/flatbuffers/releases/download/v25.12.19/Linux.flatc.binary.g++-13.zip && \
    unzip Linux.flatc.binary.g++-13.zip && \
    install flatc /usr/local/bin/flatc && \
    popd

WORKDIR /workspace/flatflow

COPY . .

# Note: Even if the wheel is successfully built, `make check` may fail in manylinux.
# This is due to the absence of Python development package and can be resolved by
# installing the relevant one such as `yum install -y python3.12-devel`.
RUN make generate && \
    make build && \
    auditwheel repair target/wheels/*

# For PyPI upload, run the commands commented out below.
# RUN pipx install twine && \
#     twine upload wheelhouse/*
