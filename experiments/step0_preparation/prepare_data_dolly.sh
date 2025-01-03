#!/bin/bash

DATA_DIR=${DATA_DIR:-"/veronica/data/dolly"}
mkdir -p ${DATA_DIR}

python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/download.py \
    --path_to_save ${DATA_DIR}

python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py \
    --input ${DATA_DIR}/databricks-dolly-15k.jsonl

python3 dataprep/split.py --input_file ${DATA_DIR}/databricks-dolly-15k-output.jsonl

echo "Dolly data is prepared under ${DATA_DIR}"
