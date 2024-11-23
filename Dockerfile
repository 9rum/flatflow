FROM quay.io/pypa/manylinux_2_28_x86_64

# This should be consistent with the gRPC version in requirements.txt.
ARG GRPC_VERSION=1.65.4

RUN echo -e "[oneAPI]\n\
name=Intel® oneAPI repository\n\
baseurl=https://yum.repos.intel.com/oneapi\n\
enabled=1\n\
gpgcheck=1\n\
repo_gpgcheck=1\n\
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB" > /etc/yum.repos.d/oneAPI.repo

RUN rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    dnf update -y && \
    dnf install -y intel-basekit && \
    dnf autoremove && \
    dnf clean all && \
    ln -s /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h /usr/include/cblas.h

WORKDIR /workspace

RUN git clone -b v${GRPC_VERSION} https://github.com/grpc/grpc.git && \
    pushd grpc && \
    git submodule update --init --recursive && \
    mkdir -p cmake/build && \
    pushd cmake/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
          -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF ../.. && \
    make -j4 && \
    make install && \
    popd && \
    popd && \
    rm -rf grpc

WORKDIR /workspace/flatflow

COPY . .

RUN source /opt/intel/oneapi/setvars.sh && \
    for PYTHON_VERSION in 3.9 3.10 3.11 3.12 3.13; \
    do \
      python$PYTHON_VERSION -m build -w && \
      rm -rf build flatflow.egg-info; \
    done && \
    auditwheel repair dist/*

# To upload to PyPI, run the commands commented out below.
# RUN pipx install twine && \
#     twine upload wheelhouse/*
