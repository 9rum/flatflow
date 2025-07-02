# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py
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

import nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset
import numpy as np
import torch
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

from flatflow.nemo.core.classes import Dataset

__all__ = ["GPTDataset"]


class GPTDataset(Dataset, nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset.GPTDataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        index_mapping_dir: Optional[str] = None,
        virtual_tokens: int = 0,
        memmap_workers: Optional[int] = None,
        truncation_method: str = "right",
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
    ) -> None:
        """
        GPT dataset for causal language modeling pretraining.
        
        file_path: Path to text files for GPT pretraining. Can be .jsonl files with 'text' field or plain .txt files.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        memmap_workers: Number of workers for memory mapping.
        truncation_method: Truncation from which position. Options: ['left', 'right']
        is_test: Whether this dataset is the test split.
        output_original_text (bool): if true, will keep the original text in the output alongside the tokenized ids.
        ceil_to_power_2: Whether to pad sequence length to power of 2.
        get_attention_mask_from_fusion: Whether to get attention mask from fusion.
        """
        nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset.GPTDataset.__init__(
            self,
            file_path,
            tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            pad_seq_length_to_mult=pad_seq_length_to_mult,
            add_bos=add_bos,
            add_eos=add_eos,
            max_num_samples=max_num_samples,  # type: ignore[arg-type]
            seed=seed,
            index_mapping_dir=index_mapping_dir,  # type: ignore[arg-type]
            virtual_tokens=virtual_tokens,
            memmap_workers=memmap_workers,
            truncation_method=truncation_method,
            is_test=is_test,
            output_original_text=output_original_text,
            ceil_to_power_2=ceil_to_power_2,
            get_attention_mask_from_fusion=get_attention_mask_from_fusion,
        )

        # Don't preprocess all data in memory for large pretraining datasets
        # Instead, process on-the-fly in __getitem__

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):  # type: ignore[arg-type]
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):  # type: ignore[arg-type]
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        auto_gen_idx = idx < 0
        if auto_gen_idx:
            idx += len(self)

        try:
            # Process the example on-the-fly for memory efficiency
            example = self._process_example(self.indexed_dataset[idx])
            
            # For causal language modeling, labels are input_ids shifted by 1
            labels = example["input_ids"][1:]
            input_ids = example["input_ids"][:-1]
            token_count = example["token_count"] - 1
            
            if auto_gen_idx:
                example["__AUTOGENERATED__"] = True
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": [0] * token_count if auto_gen_idx else [1] * token_count,
            "seqlen": token_count,
        }

    def __sizeof__(self, idx):
        if isinstance(idx, np.int64):  # type: ignore[arg-type]
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):  # type: ignore[arg-type]
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        if idx < 0:
            idx += len(self)

        try:
            example = self._process_example(self.indexed_dataset[idx])
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e

        return example["token_count"] - 1  # -1 for causal LM shift

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