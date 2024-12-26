FROM nvcr.io/nvidia/nemo:24.09

ENV WORKSPACE=/workspace

WORKDIR ${WORKSPACE}

# Below libraries are required for evaluation
RUN rm -rf ${WORKSPACE}/*
RUN pip install --upgrade pip && \
    pip install --upgrade --no-cache-dir flatflow lm-eval==0.4.5 langdetect==1.0.9 immutabledict==4.2.1

COPY . ${WORKSPACE}
