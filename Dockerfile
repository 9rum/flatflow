FROM ubuntu:22.04

ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.3.0
ARG GRPC_VERSION=1.66.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt upgrade -y && \
    apt install -y intel-mkl build-essential autoconf libtool pkg-config cmake wget zip git python${PYTHON_VERSION} python3-pip python${PYTHON_VERSION}-venv && \
    apt autopurge -y && \
    apt autoremove -y && \
    apt autoclean -y

RUN python${PYTHON_VERSION} -m pip install --upgrade pip && \
    python${PYTHON_VERSION} -m pip install build twine auditwheel patchelf

WORKDIR /workspace

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cpu.zip && \
    cp -r libtorch/include/* /usr/local/include && \
    cp -r libtorch/lib/* /usr/local/lib && \
    cp -r libtorch/share/* /usr/local/share && \
    rm -rf libtorch && \
    rm -f libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+cpu.zip

RUN git clone -b v${GRPC_VERSION} https://github.com/grpc/grpc.git && \
    cd grpc && \
    git submodule update --init --recursive && \
    mkdir -p cmake/build && \
    cd cmake/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF ../.. && \
    make -j4 && \
    make install && \
    cd /workspace && \
    rm -rf grpc

WORKDIR /workspace/flatflow
COPY . .
RUN python${PYTHON_VERSION} -m build -w
