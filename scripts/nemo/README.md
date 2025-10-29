## Prerequisites

You have to install several dependencies before running the script:

```bash
$ pip install -r scripts/nemo/requirements.txt
```

## Data preprocessing

Run the following command to process documents:

```bash
$ python3 scripts/nemo/preprocess_data_for_megatron.py \
    --input=PATH_TO_JSON_FILE \
    --json-key=JSON_KEY \
    --tokenizer-library=LIBRARY \
    --tokenizer-type=MODEL_NAME \
    --output-prefix=PATH_TO_OUTPUT_DIR \
    --max-seq-len=MAX_SEQ_LEN
```

If you preprocess [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) for Llama 3.2 1B with context length of 8192, then run the following command:

```bash
$ python3 scripts/nemo/preprocess_data_for_megatron.py \
    --input=PATH_TO_JSON_FILE \
    --json-key=content \
    --tokenizer-library=huggingface \
    --tokenizer-type=meta-llama/Llama-3.2-1B \
    --output-prefix=PATH_TO_OUTPUT_DIR \
    --max-seq-len=8192
```

> [!CAUTION]
> This script assumes that the dataset files are merged into a single .jsonl file.

This may produce a .bin file containing the indexed dataset, and its byte offsets .idx file.
