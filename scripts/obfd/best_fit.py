"""
Unofficial implementation of Fewer Truncations Improve Language Modeling [ICML '24].
"""

import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from tqdm import tqdm


def best_fit_decreasing(args):
    counts, bin_size, start = args
    items = [(start + idx, cnt) for idx, cnt in enumerate(counts)]
    items.sort(key=lambda item: item[1], reverse=True)

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


def parallel_best_fit_decreasing(counts: np.ndarray, bin_size: int):
    n_chunks = cpu_count()
    n_total = len(counts)
    chunk_size = (n_total + n_chunks - 1) // n_chunks

    tasks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_total)
        chunk = counts[start:end].copy()
        tasks.append((chunk, bin_size, start))

    pool = Pool(processes=n_chunks)
    yield from pool.imap(best_fit_decreasing, tasks)


def read_jsonl_at(path: str, index: int, offsets: np.ndarray):
    assert index < len(offsets) and offsets[index] < os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as f:
        f.seek(offsets[index])
        line = f.readline().strip()
        assert line, f"Empty line at offset {offsets[index]} for index {index}"
        return json.loads(line)


def get_tokenizer(args):
    return get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        use_fast=True,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--token-offsets", type=str, required=True)
    parser.add_argument("--label-offsets", type=str, required=True)
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
    assert os.path.exists(args.tokens), f"File does not exist: {args.tokens}"
    assert os.path.exists(args.labels), f"File does not exist: {args.labels}"
    assert os.path.exists(args.counts), f"File does not exist: {args.counts}"
    assert os.path.exists(args.token_offsets), f"File does not exist: {args.token_offsets}"
    assert os.path.exists(args.label_offsets), f"File does not exist: {args.label_offsets}"

    token_counts = np.load(args.counts)
    token_offsets = np.load(args.token_offsets)
    label_offsets = np.load(args.label_offsets)

    tokenizer = get_tokenizer(args)

    tokens_bin_file = f"{args.output_prefix}_{args.json_key}_document.bin"
    tokens_idx_file = f"{args.output_prefix}_{args.json_key}_document.idx"
    tokens_builder = indexed_dataset.make_builder(
        tokens_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    labels_bin_file = f"{args.output_prefix}_{args.json_key}_label.bin"
    labels_idx_file = f"{args.output_prefix}_{args.json_key}_label.idx"
    labels_builder = indexed_dataset.make_builder(
        labels_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    for bins in tqdm(parallel_best_fit_decreasing(token_counts, args.max_seq_len), "Iterating bins"):
        for items in tqdm(bins, desc="Saving packed sequences"):
            token_ids = []
            label_ids = []
            num_tokens = 0

            for item in items:
                num_tokens += item[1]
                data = read_jsonl_at(args.tokens, item[0], token_offsets)
                text = data[args.json_key]
                token_ids.extend(tokenizer.text_to_ids(text))
                data = read_jsonl_at(args.labels, item[0], label_offsets)
                text = data[args.json_key]
                label_ids.extend(tokenizer.text_to_ids(text))

            # pad to fit the sequence to the model's context length
            token_ids.extend([tokenizer.eos_id] * (args.max_seq_len - num_tokens))
            tokens_builder.add_item(torch.IntTensor(token_ids))
            tokens_builder.end_document()
            label_ids.extend([tokenizer.eos_id] * (args.max_seq_len - num_tokens))
            labels_builder.add_item(torch.IntTensor(label_ids))
            labels_builder.end_document()

    tokens_builder.finalize(tokens_idx_file)
    labels_builder.finalize(labels_idx_file)
    print(f"Took {time.monotonic() - now}s for best-fit packing")


if __name__ == "__main__":
    main()
