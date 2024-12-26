# Adopted from: https://docs.nvidia.com/nemo-framework/user-guide/24.07/playbooks/llama2sft.html#step-3-split-the-data-into-train-validation-and-test
import argparse
import json
import random
from pathlib import Path


def prepare_args():
    parser = argparse.ArgumentParser(description="generate dataset")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--train_out", type=str, default="train.jsonl")
    parser.add_argument("--val_out", type=str, default="val.jsonl")
    parser.add_argument("--test_out", type=str, default="test.jsonl")
    parser.add_argument("--split_ratio", type=str, default="0.95,0.05,0")
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()

args = prepare_args()
print(f"{args=}")
random.seed(args.seed)

input_file = args.input_file
output_dir = Path(args.input_file).parent
training_output_file = str(output_dir.joinpath(args.train_out))
validation_output_file = str(output_dir.joinpath(args.val_out))
test_output_file = str(output_dir.joinpath(args.test_out))

# Specify the proportion of data for training and validation
ratio_strs = args.split_ratio.split(",")
assert len(ratio_strs) == 3
train_proportion = float(ratio_strs[0])
validation_proportion = float(ratio_strs[1])
test_proportion = float(ratio_strs[2])

# Read the JSONL file and shuffle the JSON objects
with open(input_file, "r") as f:
    lines = f.readlines()
    random.shuffle(lines)

# Calculate split indices
total_lines = len(lines)
train_index = int(total_lines * train_proportion)
val_index = int(total_lines * validation_proportion)

# Distribute JSON objects into training and validation sets
train_data = lines[:train_index]
validation_data = lines[train_index:train_index+val_index]
test_data = lines[train_index+val_index:]

# Write JSON objects to training file
if train_proportion > 0:
    with open(training_output_file, "w") as f:
        for line in train_data:
            f.write(line.strip() + "\n")
    print(f"Train split of {len(train_data)} is saved into {training_output_file}")

# Write JSON objects to validation file
if validation_proportion > 0:
    with open(validation_output_file, "w") as f:
        for line in validation_data:
            f.write(line.strip() + "\n")
    print(f"Valid split of {len(validation_data)} is saved into {validation_output_file}")

# Write JSON objects to training file
if test_proportion > 0:
    with open(test_output_file, "w") as f:
        for line in test_data:
            f.write(line.strip() + "\n")
    print(f"Train split of {len(test_data)} is saved into {test_output_file}")
