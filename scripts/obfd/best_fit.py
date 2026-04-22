"""
Unofficial implementation of Fewer Truncations Improve Language Modeling [ICML '24].
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from tqdm import tqdm

_PAD_TOKEN_ID = -1


def counting_sort(items: np.ndarray, cap: int):
    now = time.monotonic()
    buckets = [[] for _ in range(cap + 1)]
    for idx, item in enumerate(items):
        buckets[item].append(idx)
    print(f"Took {time.monotonic() - now}s for counting sort\t{len(items)=}")
    return buckets


class SegmentTree(object):
    def __init__(self, cap: int):
        self._cap = cap
        self._size = 1 << (math.ceil(math.log2(cap)))
        self._tree = [0] * (self._size << 1)

    def update(self, cap: int, present: bool):
        idx = self._size + cap - 1
        self._tree[idx] = cap if present else 0
        idx >>= 1

        while idx:
            left = idx << 1
            self._tree[idx] = max(self._tree[left], self._tree[left + 1])
            idx >>= 1

    def query(self, weight: int):
        if self._tree[1] < weight:
            return 0

        idx = 1
        while idx < self._size:
            left = idx << 1
            if weight <= self._tree[left]:
                idx = left
            else:
                idx = left + 1

        return idx - self._size + 1


def optimized_best_fit_decreasing(counts: np.ndarray, bin_size: int):
    now = time.monotonic()
    buckets = counting_sort(counts, bin_size)

    def items_desc():
        for c in range(bin_size, 0, -1):
            for idx in buckets[c]:
                yield c, idx

    bins = []
    space_to_bins = {}

    tree = SegmentTree(bin_size)

    def add_bin(cap: int, bin_id: int):
        if cap not in space_to_bins:
            space_to_bins[cap] = [bin_id]
            tree.update(cap, True)
        else:
            space_to_bins[cap].append(bin_id)

    def pop_bin(cap: int):
        lst = space_to_bins[cap]
        bin_id = lst.pop()
        if not lst:
            del space_to_bins[cap]
            tree.update(cap, False)
        return bin_id

    bins.append([])
    add_bin(bin_size, 0)

    for weight, idx in items_desc():
        cap = tree.query(weight)

        if cap == 0:
            bin_id = len(bins)
            bins.append([])
            add_bin(bin_size, bin_id)
            cap = bin_size

        bin_id = pop_bin(cap)
        bins[bin_id].append(idx)

        new_cap = cap - weight
        if 0 < new_cap:
            add_bin(new_cap, bin_id)

    print(f"Took {time.monotonic() - now}s for OBFD\t{len(counts)=}")
    return bins


def get_tokenizer(args):
    return get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        use_fast=True,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-prefix", type=str, required=True)
    parser.add_argument("--label-prefix", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
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
    assert os.path.exists(args.counts), f"File does not exist: {args.counts}"

    tokenizer = get_tokenizer(args)
    chunked_tokens = indexed_dataset.make_dataset(args.token_prefix, "mmap")
    chunked_labels = indexed_dataset.make_dataset(args.label_prefix, "mmap")
    token_counts = np.load(args.counts)

    filename = os.path.basename(args.token_prefix).split("_chunked")[0]
    output_prefix = os.path.abspath(args.output_prefix)
    tokens_bin_file = os.path.join(
        output_prefix,
        f"{filename}_{args.json_key}_document.bin",
    )
    tokens_idx_file = os.path.join(
        output_prefix,
        f"{filename}_{args.json_key}_document.idx",
    )
    tokens_builder = indexed_dataset.make_builder(
        tokens_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )
    labels_bin_file = os.path.join(
        output_prefix,
        f"{filename}_{args.json_key}_label.bin",
    )
    labels_idx_file = os.path.join(
        output_prefix,
        f"{filename}_{args.json_key}_label.idx",
    )
    labels_builder = indexed_dataset.make_builder(
        labels_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    bins = optimized_best_fit_decreasing(token_counts, args.max_seq_len)

    for bin in tqdm(bins, "Saving packed sequences"):
        token_ids = []
        label_ids = []
        for idx in bin:
            token_ids.extend(chunked_tokens.get(idx))
            label_ids.extend(chunked_labels.get(idx))

        token_ids.extend([_PAD_TOKEN_ID] * (args.max_seq_len - len(token_ids)))
        tokens_builder.add_item(torch.tensor(token_ids, dtype=torch.int32))
        tokens_builder.end_document()

        label_ids.extend([_PAD_TOKEN_ID] * (args.max_seq_len - len(label_ids)))
        labels_builder.add_item(torch.tensor(label_ids, dtype=torch.int32))
        labels_builder.end_document()

    tokens_builder.finalize(tokens_idx_file)
    labels_builder.finalize(labels_idx_file)
    print(f"Took {time.monotonic() - now}s for best-fit packing")


if __name__ == "__main__":
    main()
