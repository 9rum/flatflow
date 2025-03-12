# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_dataset.py
# Copyright (c) 2024, The FlatFlow Authors.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Mapping
from typing import Optional

import nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset
import numpy as np
import torch
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

from flatflow.nemo.core.classes import Dataset

__all__ = ["GPTSFTDataset"]


class GPTSFTDataset(Dataset, nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset.GPTSFTDataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,
        index_mapping_dir: Optional[str] = None,
        prompt_template: Optional[str] = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        truncation_method: str = "right",
        special_tokens: Optional[Mapping[str, str]] = None,
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
    ) -> None:
        """
        file_path: Path to a JSONL GPT supervised fine-tuning dataset. Data is formatted as multiple JSON lines with each line formatted as follows. {'input': 'John von Neumann\nVon Neumann made fundamental contributions .... Q: What did the math of artificial viscosity do?', 'output': 'smoothed the shock transition without sacrificing basic physics'}
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        label_key: Key to use for the label in your JSONL file
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: keys in prompt_template). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {context_key}\n\nA: {label_key}
        hf_dataset: Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        is_test: Whether this dataset is the test split.
        output_original_text (bool): if true, will keep the original text in the output alongside the tokenized ids.
        """
        nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset.GPTSFTDataset.__init__(
            self,
            file_path,
            tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            pad_seq_length_to_mult=pad_seq_length_to_mult,
            add_bos=add_bos,
            add_eos=add_eos,
            add_sep=add_sep,
            sep_id=sep_id,  # type: ignore[arg-type]
            max_num_samples=max_num_samples,  # type: ignore[arg-type]
            seed=seed,
            label_key=label_key,
            answer_only_loss=answer_only_loss,
            truncation_field=truncation_field,
            pad_to_max_length=pad_to_max_length,
            index_mapping_dir=index_mapping_dir,  # type: ignore[arg-type]
            prompt_template=prompt_template,  # type: ignore[arg-type]
            virtual_tokens=virtual_tokens,
            tokens_to_generate=tokens_to_generate,
            memmap_workers=memmap_workers,
            hf_dataset=hf_dataset,
            truncation_method=truncation_method,
            special_tokens=special_tokens,
            is_test=is_test,
            output_original_text=output_original_text,
            ceil_to_power_2=ceil_to_power_2,
            get_attention_mask_from_fusion=get_attention_mask_from_fusion,
        )

        self.processed_dataset = [None] * len(self.indexed_dataset)
        for index in range(len(self.indexed_dataset)):
            self.processed_dataset[index] = self._process_example(self.indexed_dataset[index])
            self.processed_dataset[index]["labels"] = self.processed_dataset[index]["input_ids"][1:]
            self.processed_dataset[index]["input_ids"] = self.processed_dataset[index]["input_ids"][:-1]
            self.processed_dataset[index]["token_count"] -= 1

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):  # type: ignore[arg-type]
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):  # type: ignore[arg-type]
                idx = idx.item()

        assert idx < len(self.processed_dataset)
        auto_gen_idx = idx < 0
        if auto_gen_idx:
            idx += len(self)

        try:
            example = self.processed_dataset[idx]
            if auto_gen_idx:
                example["__AUTOGENERATED__"] = True
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e

        return {
            "input_ids": example["input_ids"],
            "labels": example["labels"],
            "loss_mask": [0] * example["token_count"] if auto_gen_idx else self._build_loss_mask(example),
            "seqlen": example["token_count"],
        }

    def __sizeof__(self, idx):
        if isinstance(idx, np.int64):  # type: ignore[arg-type]
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):  # type: ignore[arg-type]
                idx = idx.item()

        assert idx < len(self.processed_dataset)
        if idx < 0:
            idx += len(self)

        try:
            example = self.processed_dataset[idx]
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e

        return example["token_count"]

    def _collate_fn(self, batch):
        input_ids = np.concatenate([item["input_ids"] for item in batch])
        labels = np.concatenate([item["labels"] for item in batch])
        loss_mask = np.concatenate([item["loss_mask"] for item in batch])
        position_ids = np.concatenate([list(range(item["seqlen"])) for item in batch])
        token_count = input_ids.shape[0]

        assert input_ids.shape[0] == position_ids.shape[0]

        seqlens = np.array([item["seqlen"] for item in batch])
        cu_seqlens = np.concatenate([[0], seqlens.cumsum(), [-1]])
        cu_seqlens_argmin = np.argmin(cu_seqlens, keepdims=True)
        max_seqlen = seqlens.max(keepdims=True)

        return {
            "tokens": torch.LongTensor(input_ids).unsqueeze(0),
            "labels": torch.LongTensor(labels).unsqueeze(0),
            "loss_mask": torch.LongTensor(loss_mask).unsqueeze(0),
            "position_ids": torch.LongTensor(position_ids).unsqueeze(0),
            "token_count": [token_count],
            "attention_mask": torch.LongTensor([1]),
            "cu_seqlens": torch.IntTensor(cu_seqlens).unsqueeze(0),
            "cu_seqlens_argmin": torch.IntTensor(cu_seqlens_argmin).unsqueeze(0),
            "max_seqlen": torch.IntTensor(max_seqlen).unsqueeze(0),
        }

    def collate_fn(self, batch):
        if self.input_types is not None:
            raise TypeError("Datasets should not implement `input_types` as they are not checked")

        return self._collate_fn(batch)
