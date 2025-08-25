#!/bin/bash

# First, prepare jsonl files of `text` fields
# Then, run the script

# References
# - https://docs.nvidia.com/nemo-framework/user-guide/latest/data/pretrain_data.html

DATA_DIR=${DATA_DIR}
OUTPUT_DIR=${OUTPUT_DIR}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"meta-llama/Llama-3.1-8B"}

mkdir -p ${OUTPUT_DIR}

python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=${DATA_DIR} --preproc-folder \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=${TOKENIZER_MODEL} \
    --dataset-impl=mmap \
    --output-prefix=${OUTPUT_DIR}/ \
    --append-eod \
    --workers=48
