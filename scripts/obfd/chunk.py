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


def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
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
    fout = open(f"{args.output_prefix}{filename}_chunk.jsonl", "w", encoding="utf-8")
    token_counts = []

    for line in tqdm(fin):
        data = json.loads(line)
        text = data[args.json_key]
        ids = tokenizer.text_to_ids(text)
        ids.insert(0, tokenizer.bos_id)  # type: ignore
        ids.append(tokenizer.eos_id)  # type: ignore
        num_tokens = len(ids)

        if args.max_seq_len < num_tokens:
            # Perform per-doc chunking based on Figure 1 in `Fewer Truncations Improve
            # Language Modeling`. See https://openreview.net/pdf?id=kRxCDDFNpp.
            for idx in range(0, num_tokens, args.max_seq_len):
                chunk_ids = ids[idx : idx + args.max_seq_len]
                chunk_text = tokenizer.ids_to_text(chunk_ids)
                fout.write(
                    json.dumps({args.json_key: chunk_text}, ensure_ascii=False) + "\n"
                )
                token_counts.append(len(chunk_ids))
        else:
            text = tokenizer.ids_to_text(ids)
            fout.write(json.dumps({args.json_key: text}, ensure_ascii=False) + "\n")
            token_counts.append(num_tokens)

    fin.close()
    fout.close()
    np.save(f"{args.output_prefix}{filename}_chunk_meta.npy", np.array(token_counts))
    print(f"Took {time.monotonic() - now}s for per-doc chunking")


if __name__ == "__main__":
    main()
