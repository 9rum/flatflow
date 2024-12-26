#!/bin/bash

# A temporary script to setup debugging environment easily, to be removed later.

pip install flatflow 

cp megatron_gpt_finetuning_config.yaml /opt/NeMo/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml
cp megatron_gpt_finetuning.py /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py

rm /usr/local/lib/python3.10/dist-packages/flatflow/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py
ln -s /shared_workspace/project/fork_flatflow/flatflow/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py /usr/local/lib/python3.10/dist-packages/flatflow/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py

rm /usr/local/lib/python3.10/dist-packages/flatflow/nemo/collections/nlp/models/language_modeling/megatron_gpt_sft_model.py
ln -s /shared_workspace/project/fork_flatflow/flatflow/nemo/collections/nlp/models/language_modeling/megatron_gpt_sft_model.py /usr/local/lib/python3.10/dist-packages/flatflow/nemo/collections/nlp/models/language_modeling/megatron_gpt_sft_model.py
