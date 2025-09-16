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
    tokens = tokenizer.ids_to_tokens(ids)
    text = tokenizer.tokens_to_text(tokens)
    return text


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
    fout = open(f"{args.output_prefix}{filename}_chunk.jsonl", "wb")

    max_seq_len = args.max_seq_len + 1
    token_counts = []
    offsets = [0]

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

        ids.insert(0, tokenizer.bos_id)
        ids.append(tokenizer.eos_id)
        num_tokens = len(ids)

        if max_seq_len < num_tokens:
            # Perform per-doc chunking based on Figure 1 in `Fewer Truncations Improve
            # Language Modeling`. See https://openreview.net/pdf?id=kRxCDDFNpp.
            for idx in range(0, num_tokens, max_seq_len):
                chunk_ids = ids[idx : idx + max_seq_len]
                chunk_text = ids_to_text(tokenizer, chunk_ids)
                s = json.dumps({args.json_key: chunk_text}, ensure_ascii=False) + "\n"
                s = s.encode()
                fout.write(s)
                token_counts.append(len(chunk_ids))
                offsets.append(offsets[-1] + len(s))
        else:
            text = ids_to_text(tokenizer, ids)
            s = json.dumps({args.json_key: text}, ensure_ascii=False) + "\n"
            s = s.encode()
            fout.write(s)
            token_counts.append(num_tokens)
            offsets.append(offsets[-1] + len(s))

    fin.close()
    fout.close()
    np.save(f"{args.output_prefix}{filename}_chunk_cnt.npy", np.array(token_counts))
    np.save(f"{args.output_prefix}{filename}_chunk_idx.npy", np.array(offsets))
    print(f"Took {time.monotonic() - now}s for per-doc chunking")


if __name__ == "__main__":
    main()
