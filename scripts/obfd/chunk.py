"""
Unofficial implementation of Fewer Truncations Improve Language Modeling [ICML '24].
"""

import argparse
import json
import os
import time

import numpy as np
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from tqdm import tqdm


def ids_to_text(tokenizer, ids):
    return tokenizer.tokens_to_text(tokenizer.ids_to_tokens(ids))


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
    filename = os.path.splitext(os.path.basename(args.input))[0]
    fin = open(args.input, "r", encoding="utf-8")
    fout_tokens = open(f"{args.output_prefix}{filename}_tokens.jsonl", "wb")
    fout_labels = open(f"{args.output_prefix}{filename}_labels.jsonl", "wb")

    token_counts = []
    token_offsets = [0]
    label_offsets = [0]

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

        if args.max_seq_len < num_tokens:
            # Perform per-doc chunking based on Figure 1 in `Fewer Truncations Improve
            # Language Modeling`. See https://openreview.net/pdf?id=kRxCDDFNpp.
            for idx in range(0, num_tokens, args.max_seq_len):
                token_chunk_ids = token_ids[idx : idx + args.max_seq_len]
                token_chunk_text = ids_to_text(tokenizer, token_chunk_ids)
                s = (
                    json.dumps({args.json_key: token_chunk_text}, ensure_ascii=False)
                    + "\n"
                )
                s = s.encode()
                fout_tokens.write(s)
                token_offsets.append(token_offsets[-1] + len(s))
                token_counts.append(len(token_chunk_ids))

                label_chunk_ids = label_ids[idx : idx + args.max_seq_len]
                label_chunk_text = ids_to_text(tokenizer, label_chunk_ids)
                s = (
                    json.dumps({args.json_key: label_chunk_text}, ensure_ascii=False)
                    + "\n"
                )
                s = s.encode()
                fout_labels.write(s)
                label_offsets.append(label_offsets[-1] + len(s))
        else:
            token_text = ids_to_text(tokenizer, token_ids)
            s = json.dumps({args.json_key: token_text}, ensure_ascii=False) + "\n"
            s = s.encode()
            fout_tokens.write(s)
            token_offsets.append(token_offsets[-1] + len(s))
            token_counts.append(num_tokens)

            label_text = ids_to_text(tokenizer, label_ids)
            s = json.dumps({args.json_key: label_text}, ensure_ascii=False) + "\n"
            s = s.encode()
            fout_labels.write(s)
            label_offsets.append(label_offsets[-1] + len(s))

    fin.close()
    fout_tokens.close()
    fout_labels.close()
    np.save(f"{args.output_prefix}{filename}_cnt.npy", np.array(token_counts))
    np.save(f"{args.output_prefix}{filename}_token_idx.npy", np.array(token_offsets))
    np.save(f"{args.output_prefix}{filename}_label_idx.npy", np.array(label_offsets))
    print(f"Took {time.monotonic() - now}s for per-doc chunking")


if __name__ == "__main__":
    main()
