#!/bin/bash

# A temporary script to setup debugging environment easily, to be removed later.
# Run: bash experiments/init.sh     # Mind the working directory.

pip install flatflow

cp {examples/nemo,/opt/NeMo/examples}/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml
cp {examples/nemo,/opt/NeMo/examples}/nlp/language_modeling/tuning/megatron_gpt_finetuning.py
cp {examples/nemo,/opt/NeMo/examples}/nlp/language_modeling/megatron_gpt_pretraining.py
cp {examples/nemo,/opt/NeMo/examples}/nlp/language_modeling/conf/llama3_1_8b.yaml
