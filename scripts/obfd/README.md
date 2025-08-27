This is a naive implementation of [Fewer Truncations Improve Language Modeling](https://proceedings.mlr.press/v235/ding24f.html) [ICML '24].

## Prerequisites

You have to install several dependencies before running the scripts:

```bash
$ pip install -r scripts/obfd/requirements.txt
```

## Per-doc chunking

The proposed approach requires per-doc chunking before document packing.
Run the following command to perform per-doc chunking:

```bash
$ python3 scripts/obfd/chunk.py \
    --input=PATH_TO_JSON_FILE \
    --json-key=JSON_KEY \
    --tokenizer-library=LIBRARY \
    --tokenizer-type=MODEL_NAME \
    --output-prefix=PATH_TO_OUTPUT_DIR \
    --max-seq-len=MAX_SEQ_LEN
```

If you preprocess [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) for Llama 3.2 1B with context length of 8192, then run the following command:

```bash
$ python3 scripts/obfd/chunk.py \
    --input=PATH_TO_JSON_FILE \
    --json-key=content \
    --tokenizer-library=huggingface \
    --tokenizer-type=meta-llama/Llama-3.2-1B \
    --output-prefix=PATH_TO_OUTPUT_DIR \
    --max-seq-len=8192
```

> [!CAUTION]
> This script assumes that the dataset files are merged into a single .jsonl file.

This may produce a .jsonl file containing the chunked documents, and its metadata .npy file.

## Best-fit packing
