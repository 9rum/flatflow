#!/bin/bash

DATA_DIR=${DATA_DIR:-"/veronica/data/alpaca"}
mkdir -p ${DATA_DIR}

python3 dataprep/download_and_process_alpaca.py --output-dir ${DATA_DIR}

echo "Alpaca data is prepared under ${DATA_DIR}"
