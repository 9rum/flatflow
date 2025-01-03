# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script to process the Anthropic Dataset"""
import argparse
import json
import random
import warnings
from pathlib import Path

from datasets import load_dataset


def prepare_args():
    parser = argparse.ArgumentParser(description="generate dataset")
    parser.add_argument("--data_dir", type=str)
    return parser.parse_args()


START_PROMPT_FORMAT = "User: {body}\n\nAssistant: {response}"
PROMPT_CONTINUATION_FORMAT = "{text}\n\nUser: {body}\n\nAssistant: {response}"


# Adopted and modified from: https://github.com/zphang/lrqa/blob/main/lrqa/preproc/simple.py 
def process_raw_data(raw_data):
    out = []
    for row in raw_data:
        context = row["article"]
        for idx, question_dict in enumerate(row["questions"]):
            qa_item = {
                "context": "".join(context),
                "query": " " + question_dict[f"question"].strip(),
                "label": question_dict[f"gold_label"],
                "options": [option.strip() for option in question_dict["options"]],
            }
            out.append(qa_item)
    return out


def process_quality(data_dir, split):
    file_path = f"{data_dir}/raw_{split}.jsonl"
    with open(file_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]
    lines = process_raw_data(lines)

    list_of_dicts = []
    for item in lines:
        input = f"### Context: {item['context']}\n\n### Query: {item['query']}\n\n### Options\n\n"
        input += "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(item["options"])])
        output = f"### Response: {item['label']}"
        instruction_dict = {
            "input": input,
            "output": output,
        }
        list_of_dicts.append(instruction_dict)

    return list_of_dicts


def convert_list_of_dict_to_jsonl(list_of_dict):
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in list_of_dict)


def save_dataset(list_of_dict, save_dir, split="train"):
    with open(f"{save_dir}/{split}.jsonl", "w", encoding="utf-8") as f:
        f.write(convert_list_of_dict_to_jsonl(list_of_dict))


if __name__ == "__main__":
    args = prepare_args()

    for split in ["train", "val"]:
        list_of_dicts = process_quality(args.data_dir, split)
        save_dataset(list_of_dicts, args.data_dir, split)
