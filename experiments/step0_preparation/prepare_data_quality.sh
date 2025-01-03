#!/bin/bash

DATA_DIR=${DATA_DIR:-"/veronica/data/quality"}
mkdir -p ${DATA_DIR}

TRAIN_DATA="https://raw.githubusercontent.com/nyu-mll/quality/refs/heads/main/data/v1.0/QuALITY.v1.0.htmlstripped.train"
VALID_DATA="https://raw.githubusercontent.com/nyu-mll/quality/refs/heads/main/data/v1.0/QuALITY.v1.0.htmlstripped.dev"
TEST_DATA="https://raw.githubusercontent.com/nyu-mll/quality/refs/heads/main/data/v1.0/QuALITY.v1.0.htmlstripped.test"

TRAIN_PATH="${DATA_DIR}/raw_train.jsonl"
VALID_PATH="${DATA_DIR}/raw_train.jsonl"
TEST_PATH="${DATA_DIR}/raw_train.jsonl"

if [ ! -f ${TRAIN_PATH} ]; then
    wget ${TRAIN_DATA} -O ${TRAIN_PATH}
fi
if [ ! -f ${VALID_PATH} ]; then
    wget ${VALID_DATA} -O ${VALID_PATH}
fi
if [ ! -f ${TEST_PATH} ]; then
    wget ${TEST_DATA} -O ${TEST_PATH}
fi

python3 dataprep/process_quality.py --data_dir ${DATA_DIR}

echo "Quality data is prepared under ${DATA_DIR}"
