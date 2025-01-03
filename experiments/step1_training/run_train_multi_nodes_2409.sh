#!/bin/bash

# Reference: https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/sft.html#id1
# Usages
# - RANK=0 NNODES=2 BATCH_SIZE=16 MICRO_BATCH_SIZE=4 PP_SIZE=4 MODEL_PATH=meta-llama/Llama-3.1-8B DATASET_NAME=quality HF_TOKEN={TOKEN} WANDB_API_KEY={KEY} bash run_train_multi_nodes.sh
# - RANK=1 NNODES=2 BATCH_SIZE=16 MICRO_BATCH_SIZE=4 PP_SIZE=4 MODEL_PATH=meta-llama/Llama-3.1-8B DATASET_NAME=quality HF_TOKEN={TOKEN} WANDB_API_KEY={KEY} bash run_train_multi_nodes.sh

MODELS_DIR=${MODELS_DIR:-"/veronica/models"}
MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-3.1-70B"}
NEMO_MODEL="${MODELS_DIR}/${MODEL_PATH}/model.nemo"
DATASET_NAME=${DATASET_NAME:-"dolly"}

DATA_DIR=${DATA_DIR:-"/veronica/data/${DATASET_NAME}"}
TRAIN_DATA="${DATA_DIR}/train.jsonl"
VALID_DATA="${DATA_DIR}/val.jsonl"

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

export NCCL_SOCKET_IFNAME=eno3
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RANK=${RANK:-0}

NNODES=${NNODES:-1}
RUN_NAME="${MODEL_PATH}/${DATASET_NAME}_TP${TP_SIZE}_PP${PP_SIZE}_BS${BATCH_SIZE}_MBS${MICRO_BATCH_SIZE}_NNODES${NNODES}"

echo "Start training: ${RUN_NAME}"
echo "Multi-node setting: ${MASTER_ADDR}:${MASTER_PORT} (rank: ${RANK}/${NNODES})"
torchrun --nproc-per-node 8 --nnodes ${NNODES} --node-rank ${RANK} \
    --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_sft.py \
       trainer.precision=bf16 \
       trainer.num_nodes=${NNODES} \
       trainer.devices=8 \
       trainer.sft.max_steps=-1 \
       trainer.sft.limit_val_batches=40 \
       trainer.sft.val_check_interval=-1 \
       trainer.sft.save_interval=10000000 \
       model.megatron_amp_O2=True \
       model.restore_from_path=${NEMO_MODEL} \
       model.answer_only_loss=True \
       model.tensor_model_parallel_size=${TP_SIZE} \
       model.pipeline_model_parallel_size=${PP_SIZE} \
       model.optim.lr=5e-6 \
       model.data.num_workers=0 \
       model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
       model.data.train_ds.global_batch_size=${BATCH_SIZE} \
       model.data.train_ds.max_seq_length=8192 \
       model.data.train_ds.file_path=${VALID_DATA} \
       model.data.validation_ds.micro_batch_size=1 \
       model.data.validation_ds.global_batch_size=${BATCH_SIZE} \
       model.data.validation_ds.file_path=${VALID_DATA} \
       model.data.validation_ds.max_seq_length=8192\
       exp_manager.create_wandb_logger=False \
       exp_manager.wandb_logger_kwargs.project=MLE-veronica \
       exp_manager.wandb_logger_kwargs.name=${RUN_NAME} \
       exp_manager.explicit_log_dir="/veronica/results/${RUN_NAME}" \
       exp_manager.resume_if_exists=True \
       exp_manager.resume_ignore_no_checkpoint=True \
       exp_manager.create_checkpoint_callback=False \
       exp_manager.checkpoint_callback_params.monitor=validation_loss \
       exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \


echo "Nemo training is done: /veronica/results/${RUN_NAME}"
