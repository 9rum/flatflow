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

This may produce .jsonl files containing the chunked tokens and labels, and its metadata .npy files.

## Best-fit packing

You may obtain .jsonl files containing the chunked tokens and labels, and three .npy files.
The file that ends with `_cnt.npy` contains token counts, and the files ending in `_idx.npy` contain byte offsets.
Run the following command to perform document packing:

```bash
$ python3 scripts/obfd/best_fit.py \
    --tokens=PATH_TO_TOKENS_JSON_FILE \
    --labels=PATH_TO_LABELS_JSON_FILE \
    --counts=PATH_TO_COUNTS_FILE \
    --token-offsets=PATH_TO_TOKEN_OFFSETS_FILE \
    --label-offsets=PATH_TO_LABEL_OFFSETS_FILE \
    --json-key=JSON_KEY \
    --tokenizer-library=LIBRARY \
    --tokenizer-type=MODEL_NAME \
    --output-prefix=PATH_TO_OUTPUT_DIR \
    --max-seq-len=MAX_SEQ_LEN
```

For the example above, run the following command:

```bash
$ python3 scripts/obfd/best_fit.py \
    --tokens=PATH_TO_TOKENS_JSON_FILE \
    --labels=PATH_TO_LABELS_JSON_FILE \
    --counts=PATH_TO_COUNTS_FILE \
    --token-offsets=PATH_TO_TOKEN_OFFSETS_FILE \
    --label-offsets=PATH_TO_LABEL_OFFSETS_FILE \
    --json-key=content \
    --tokenizer-library=huggingface \
    --tokenizer-type=meta-llama/Llama-3.2-1B \
    --output-prefix=PATH_TO_OUTPUT_DIR \
    --max-seq-len=8192
```

This may produce .bin files containing the packed sequences, and its byte offsets .idx files.
