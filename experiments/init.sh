#!/bin/bash

# A temporary script to setup debugging environment easily, to be removed later.
# Run: bash experiments/init.sh

pip install flatflow==0.0.11

cp {examples/nemo,/opt/NeMo/examples}/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml
cp {examples/nemo,/opt/NeMo/examples}/nlp/language_modeling/tuning/megatron_gpt_finetuning.py
