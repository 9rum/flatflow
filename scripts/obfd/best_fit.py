"""
Unofficial implementation of Fewer Truncations Improve Language Modeling [ICML '24].
"""

import argparse
import os
import time

import numpy as np


def best_fit_decreasing(counts: np.ndarray, bin_size: int):
    items = [(idx, cnt) for idx, cnt in enumerate(counts)]
    items.sort(key=lambda x: x[1], reverse=True)

    bins = []
    capacities = []

    for item in items:
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--json-key", type=str, default="content")
    parser.add_argument(
        "--tokenizer-library",
        type=str,
        default="huggingface",
        choices=["sentencepiece", "megatron", "huggingface", "tabular"],
    )
    parser.add_argument("--tokenizer-type", type=str, required=True)
    parser.add_argument("--need-pad-id", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--output-prefix", type=str, required=True)
    return parser.parse_args()


def main():
    now = time.monotonic()

    args = get_args()
    assert os.path.exists(args.input), f"File does not exist: {args.input}"
    assert os.path.exists(args.meta), f"File does not exist: {args.meta}"

    token_counts = np.load(args.meta)
    bins = best_fit_decreasing(token_counts, args.max_seq_len)

    print(f"Took {time.monotonic() - now}s for best-fit packing")


if __name__ == "__main__":
    main()
