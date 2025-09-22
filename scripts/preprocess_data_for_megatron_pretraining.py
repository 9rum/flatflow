import argparse
import json
import os
import time

import torch
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from tqdm import tqdm


def get_tokenizer(args):
    return get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        use_fast=True,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--json-key", type=str, default="content")
    parser.add_argument(
        "--tokenizer-library",
        type=str,
        default="huggingface",
        choices=["sentencepiece", "megatron", "huggingface", "tabular"],
    )
    parser.add_argument("--tokenizer-type", type=str, required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--output-prefix", type=str, required=True)
    return parser.parse_args()


def main():
    now = time.monotonic()

    args = get_args()
    assert os.path.exists(args.input), f"File does not exist: {args.input}"

    tokenizer = get_tokenizer(args)

    output_bin_file = f"{args.output_prefix}_{args.json_key}_document.bin"
    output_idx_file = f"{args.output_prefix}_{args.json_key}_document.idx"
    builder = indexed_dataset.make_builder(
        output_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing file {args.input}"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = data[args.json_key]
            if not text:
                continue
            ids = tokenizer.text_to_ids(text)
            if not ids:
                continue

            ids = [tokenizer.bos_id, *ids, tokenizer.eos_id]
            for idx in range(0, len(ids), args.max_seq_len):
                chunk_ids = ids[idx : idx + args.max_seq_len + 1]
                if 1 < len(chunk_ids):
                    builder.add_item(torch.IntTensor(chunk_ids))
                    builder.end_document()

    builder.finalize(output_idx_file)
    print(f"Took {time.monotonic() - now}s for processing file {args.input}")


if __name__ == "__main__":
    main()
