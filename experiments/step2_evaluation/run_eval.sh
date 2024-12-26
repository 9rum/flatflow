#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-3.1-70B"}

torchrun --nproc-per-node=8 --no-python lm_eval  \
    --model nemo_lm \
    --model_args path=${MODEL_PATH},devices=8,tensor_model_parallel_size=1,pipeline_model_parallel_size=8 \
    --batch_size 16 \
    --tasks ifeval,arc_easy,arc_challenge \
    --output_path ${MODEL_PATH}/eval_results.txt --log_samples \
    --trust_remote_code \
    # --wandb_args project=MLE-veronica,job_type=eval \
