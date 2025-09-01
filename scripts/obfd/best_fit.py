"""
Unofficial implementation of Fewer Truncations Improve Language Modeling [ICML '24].
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from tqdm import tqdm


def best_fit_decreasing(counts: np.ndarray, bin_size: int):
    items = [(idx, cnt) for idx, cnt in enumerate(counts)]
    items.sort(key=lambda x: x[1], reverse=True)

    bins = []
    capacities = []

    for item in tqdm(items, desc="Running BFD"):
        best_bin_idx = -1
        min_remaining = bin_size + 1

        for i in range(len(bins)):
            if item[1] <= capacities[i] and capacities[i] - item[1] < min_remaining:
                best_bin_idx = i
                min_remaining = capacities[i] - item[1]

        if best_bin_idx != -1:
            bins[best_bin_idx].append(item)
            capacities[best_bin_idx] -= item[1]
        else:
            bins.append([item])
            capacities.append(bin_size - item[1])

    return bins


def read_jsonl_at(file: str, index: int, offsets: np.ndarray):
    f = open(file, "r", encoding="utf-8")
    f.seek(offsets[index])
    return json.loads(f.readline())


def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        use_fast=True,
    )
    if (
        not hasattr(tokenizer, "pad_id")
        or tokenizer.pad_id is None  # type: ignore
        or tokenizer.pad_id < 0  # type: ignore
    ):
        tokenizer.add_special_tokens({"pad_token": "<pad>"})  # type: ignore
    return tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--offsets", type=str, required=True)
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
    assert os.path.exists(args.counts), f"File does not exist: {args.counts}"
    assert os.path.exists(args.offsets), f"File does not exist: {args.offsets}"

    max_seq_len = args.max_seq_len
    token_counts = np.load(args.counts)
    offsets = np.load(args.offsets)
    bins = best_fit_decreasing(token_counts, max_seq_len)

    tokenizer = get_tokenizer(args)

    output_bin_file = f"{args.output_prefix}_{args.json_key}_document.bin"
    output_idx_file = f"{args.output_prefix}_{args.json_key}_document.idx"
    builder = indexed_dataset.make_builder(
        output_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    for items in tqdm(bins, desc="Saving packed sequences"):
        ids = []
        num_tokens = 0

        for item in items:
            data = read_jsonl_at(args.input, item[0], offsets)
            text = data[args.json_key]
            chunk_ids = tokenizer.text_to_ids(text)
            ids.extend(chunk_ids)
            num_tokens += item[1]

        # pad to fit the sequence to the model's context length
        ids.extend([tokenizer.pad_id] * (max_seq_len - num_tokens))  # type: ignore
        builder.add_item(torch.IntTensor(ids))
        builder.end_document()

    builder.finalize(output_idx_file)
    print(f"Took {time.monotonic() - now}s for best-fit packing")


if __name__ == "__main__":
    main()
