#!/bin/bash

# https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/sft.html#obtain-a-pretrained-model

MODELS_DIR=${MODELS_DIR:-"/veronica/models"}
MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-3.1-70B"}

if [ ! -e "${MODELS_DIR}/${MODEL_PATH}" ]; then
    if [ ! -n ${TOKEN} ]; then
        echo "use TOKEN={your_huggingface_token} bash ..."
    else
        python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer;model = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}', token='${TOKEN}');model.save_pretrained('${MODELS_DIR}/${MODEL_PATH}');tokenizer = AutoTokenizer.from_pretrained('${MODEL_PATH}', token='${TOKEN}');tokenizer.save_pretrained('${MODELS_DIR}/${MODEL_PATH}')"
    fi
fi

# Below process takes about 40mins on a DGX-H100.
if [ -e "${MODELS_DIR}/${MODEL_PATH}" ]; then
    python3 /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
        --input_name_or_path "${MODELS_DIR}/${MODEL_PATH}" --output_path "${MODELS_DIR}/${MODEL_PATH}/model.nemo"
else
    echo "The model path '${MODELS_DIR}/${MODEL_PATH}' does not exist!"
fi
