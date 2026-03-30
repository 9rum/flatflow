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

    print(f"Processing file {args.input}")
    fin = open(args.input, "r", encoding="utf-8")
    filename, _ = os.path.splitext(os.path.basename(args.input))
    output_prefix = os.path.abspath(args.output_prefix)

    tokens_bin_file = os.path.join(output_prefix, f"{filename}_chunked_tokens.bin")
    tokens_idx_file = os.path.join(output_prefix, f"{filename}_chunked_tokens.idx")
    tokens_builder = indexed_dataset.make_builder(
        tokens_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    labels_bin_file = os.path.join(output_prefix, f"{filename}_chunked_labels.bin")
    labels_idx_file = os.path.join(output_prefix, f"{filename}_chunked_labels.idx")
    labels_builder = indexed_dataset.make_builder(
        labels_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    token_counts = []

    for line in tqdm(fin, desc="Chunking documents"):
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

        token_ids = [tokenizer.bos_id, *ids]
        label_ids = [*ids, tokenizer.eos_id]
        num_tokens = len(token_ids)

        # Perform per-doc chunking based on Figure 1 in `Fewer Truncations Improve
        # Language Modeling`. See https://openreview.net/pdf?id=kRxCDDFNpp.
        for idx in range(0, num_tokens, args.max_seq_len):
            token_chunk_ids = token_ids[idx : idx + args.max_seq_len]
            tokens_builder.add_item(torch.tensor(token_chunk_ids, dtype=torch.int32))
            tokens_builder.end_document()
            label_chunk_ids = label_ids[idx : idx + args.max_seq_len]
            labels_builder.add_item(torch.tensor(label_chunk_ids, dtype=torch.int32))
            labels_builder.end_document()
            token_counts.append(len(token_chunk_ids))

    fin.close()
    tokens_builder.finalize(tokens_idx_file)
    labels_builder.finalize(labels_idx_file)
    np.save(os.path.join(output_prefix, f"{filename}_cnt.npy"), np.array(token_counts))
    print(f"Took {time.monotonic() - now}s for per-doc chunking")


if __name__ == "__main__":
    main()
