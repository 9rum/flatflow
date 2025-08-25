#!/bin/bash

# Reference: https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2sft.html#supervised-fine-tuning-playbook
# Usages
# - USE_FLATFLOW=True MASTER_ADDR={MASTER_ADDR} MASTER_PORT={MASTER_PORT} RANK=0 NNODES=2 BATCH_SIZE=16 MICRO_BATCH_SIZE=4 PP_SIZE=4 MODEL_PATH=meta-llama/Llama-3.1-8B DATASET_NAME=quality HF_TOKEN={TOKEN} WANDB_API_KEY={KEY} bash run_pretrain_multi_nodes.sh
# - USE_FLATFLOW=True MASTER_ADDR={MASTER_ADDR} MASTER_PORT={MASTER_PORT} RANK=1 NNODES=2 BATCH_SIZE=16 MICRO_BATCH_SIZE=4 PP_SIZE=4 MODEL_PATH=meta-llama/Llama-3.1-8B DATASET_NAME=quality HF_TOKEN={TOKEN} WANDB_API_KEY={KEY} bash run_pretrain_multi_nodes.sh

MODELS_DIR=${MODELS_DIR:-"/veronica/models"}

DATA_DIR=${DATA_DIR}
DATA_PREFIX="${DATA_DIR}/_text_document"
CONFIG_NAME="llama3_1_8b"

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

USE_FLATFLOW=${USE_FLATFLOW:-False}

# export NCCL_SOCKET_IFNAME=eno3
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=${RANK:-0}

NNODES=${NNODES:-1}
RUN_NAME="${MODEL_PATH}/${DATASET_NAME}_TP${TP_SIZE}_PP${PP_SIZE}_BS${BATCH_SIZE}_MBS${MICRO_BATCH_SIZE}_NNODES${NNODES}_FLATFLOW-${USE_FLATFLOW}"

echo "Start training from scratch with ${CONFIG_NAME} on ${TRAIN_DATA}"
echo "Multi-node setting: ${MASTER_ADDR}:${MASTER_PORT} (rank: ${RANK}/${NNODES})"
torchrun --nproc-per-node 4 --nnodes ${NNODES} --node-rank ${RANK} \
    --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
       --config-path conf/ --config-name ${CONFIG_NAME} \
       trainer.devices=4 \
       trainer.num_nodes=${NNODES} \
       trainer.max_steps=100 \
       trainer.max_epochs=1 \
       trainer.val_check_interval=1.0 \
       model.use_flatflow=${USE_FLATFLOW} \
       model.micro_batch_size=${MICRO_BATCH_SIZE} \
       model.global_batch_size=${BATCH_SIZE} \
       model.tensor_model_parallel_size=${TP_SIZE} \
       model.pipeline_model_parallel_size=${PP_SIZE} \
       model.context_parallel_size=${CP_SIZE} \
       model.virtual_pipeline_model_parallel_size=1 \
       model.data.data_prefix=[1.0,${DATA_PREFIX}] \
       exp_manager.create_wandb_logger=False \
       exp_manager.explicit_log_dir="/veronica/results/${RUN_NAME}" \
       exp_manager.resume_if_exists=True \
       exp_manager.resume_ignore_no_checkpoint=True \
       exp_manager.create_checkpoint_callback=True \
       exp_manager.checkpoint_callback_params.monitor=validation_loss \
       exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
       ++cluster_type=BCP

echo "Nemo training is done: /veronica/results/${RUN_NAME}"
