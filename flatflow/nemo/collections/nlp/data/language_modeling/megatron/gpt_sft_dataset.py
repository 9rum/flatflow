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
from typing import Optional, Literal

import numpy as np
import torch
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset as NeMoGPTSFTDataset
from nemo.utils import logging

from flatflow.nemo.core.classes import Dataset

__all__ = ["GPTSFTDataset"]

MESSAGES = "messages"
ROLE = "role"
CONTENT = "content"
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

class GPTSFTDataset(Dataset, NeMoGPTSFTDataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        max_prompt_length: int = 512,
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
        truncation_method: Literal["left", "right"] = "right",
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
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.max_prompt_length = max_prompt_length
        self.pad_seq_length_to_mult = pad_seq_length_to_mult
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.label_key = label_key
        self.answer_only_loss = answer_only_loss
        self.truncation_fields = truncation_field.split(',') if truncation_field is not None else []
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.memmap_workers = memmap_workers
        self.hf_dataset = hf_dataset
        self.truncation_method = truncation_method
        self.is_test = is_test
        self.output_original_text = output_original_text
        self.ceil_to_power_2 = ceil_to_power_2
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion
        self.global_sample_mapping = False
        self.sanity_check_dist_workers = True
        self.special_tokens = special_tokens
        self.has_chat_template = hasattr(tokenizer, "apply_chat_template")
        
        NeMoGPTSFTDataset._load_dataset(self)

        if not self.has_chat_template:
            NeMoGPTSFTDataset._maybe_validate_prompt_template(self)

        if max_prompt_length >= max_seq_length:
            raise ValueError(f"max_prompt_length ({max_prompt_length}) must be < max_seq_length ({max_seq_length})")
            
        NeMoGPTSFTDataset._build_samples_mapping(self)
        
        """
        NOTE: The `token_count` is used by FlatFlow scheduler via `__sizeof__` before sampling and collation.
        So, the 'teacher-forcing alignment' should happen here, before `__sizeof__`, in contrast to `NeMoGPTSFPTDataset`
        where it happens inside `collate_fn`.

        Re-calculation of `token_count`, not simply imposing `-= 1`, is to support `GPTSFTChatDataset` whose
        `_process_example` doesn't set it.
        """
        self.processed_dataset = [None] * len(self.indexed_dataset)
        for index in range(len(self.indexed_dataset)):
            self.processed_dataset[index] = self._process_example(self.indexed_dataset[index])
            self.processed_dataset[index]["labels"] = self.processed_dataset[index]["input_ids"][1:]
            self.processed_dataset[index]["input_ids"] = self.processed_dataset[index]["input_ids"][:-1]
            self.processed_dataset[index]["token_count"] = len(self.processed_dataset[index]["input_ids"])

    def _process_example(self, example):
        if not self.has_chat_template:
            return NeMoGPTSFTDataset._process_example(self, example)
        
        return self._process_with_chat_template(example)
    
    def _process_with_chat_template(self, example):
        # normalize to {"messages": [...]}
        example = self._ensure_example_format(example)
        
        # validate basic conversation structure (SYSTEM optional, ends with ASSISTANT, etc.)
        if not self._validate_example(example):
            raise ValueError("Example validation failed: the conversation does not follow the required structure or contains invalid content.")
        
        msgs = example[MESSAGES]
        
        # truncate if over length limits
        msgs = self._truncate_example(msgs)
        
        # apply the chat template to messages
        user_ids = self.tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_tensors=None)
        input_ids = self.tokenizer.apply_chat_template(msgs, return_tensors=None)
        if self.add_eos and self.tokenizer.eos_id is not None and input_ids[-1] != self.tokenizer.eos_id:
            if len(input_ids) < self.max_seq_length:
                input_ids.append(self.tokenizer.eos_id)
            else:
                input_ids[-1] = self.tokenizer.eos_id
        answer_ids = input_ids[len(user_ids):]
        
        metadata = {k: v for k, v in example.items() if k != MESSAGES}
        if self.output_original_text:
            for msg in example[MESSAGES]:
                metadata[msg[ROLE]] = msg[CONTENT]
                
        processed_example={
            "input_ids": input_ids,
            "answer_start_idx": len(user_ids),
            "context_ids": user_ids,
            "context_length": len(user_ids),
            "answer_ids": answer_ids,
            "metadata": metadata,
            "token_count": len(input_ids)
        }
        
        return processed_example
    
    def _truncate_example(self, msgs):
        
        prompt_msgs = msgs[:-1]
        
        prompt_tpl_ids = self.tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, return_tensors=None)
        full_tpl_ids = self.tokenizer.apply_chat_template(msgs, return_tensors=None)
        
        # skip truncation if prompt and full are within limits
        if (len(prompt_tpl_ids) <= self.max_prompt_length) and (len(full_tpl_ids) <= self.max_seq_length):
            return msgs
        
        # truncation method: "right" keeps first_k, "left" keeps last_k
        if self.truncation_method not in ("right", "left"):
            raise ValueError(f"Unsupported truncation_method: {self.truncation_method}")
        first_k = lambda seq, k: (seq[:k] if k > 0 else [])
        last_k  = lambda seq, k: (seq[-k:] if k > 0 else [])
        _truncate = first_k if self.truncation_method == "right" else last_k
        
        # raw token ids (system, user, and assistant)
        has_system = prompt_msgs[0].get(ROLE) == SYSTEM
        sys_raw_ids = self.tokenizer.text_to_ids(prompt_msgs[0][CONTENT]) if has_system else []
        user_raw_ids = self.tokenizer.text_to_ids(prompt_msgs[-1][CONTENT])
        answer_raw_ids = self.tokenizer.text_to_ids(msgs[-1][CONTENT])
        
        # raw token lengths (system, user, and assistant)
        sys_raw_len = len(sys_raw_ids)
        user_raw_len = len(user_raw_ids)
        answer_raw_len = len(answer_raw_ids)
        
        # template overhead
        prompt_tpl_budget = len(prompt_tpl_ids) - (sys_raw_len + user_raw_len)
        answer_tpl_budget = (len(full_tpl_ids) - len(prompt_tpl_ids)) - answer_raw_len
        
        # allocate prompt budget: system first, then user
        prompt_budget = max(0, self.max_prompt_length - prompt_tpl_budget)
        sys_keep = min(sys_raw_len, prompt_budget)
        user_keep = min(user_raw_len, max(0, prompt_budget - sys_keep))

        # truncate system
        if has_system:
            sys_ids = _truncate(sys_raw_ids, sys_keep)
            if sys_keep < sys_raw_len:
                logging.warning(f"The system message content has been cut off by {self.truncation_method} truncation.")
            prompt_msgs[0][CONTENT] = self.tokenizer.ids_to_text(sys_ids)
        
        # truncate user
        user_ids = _truncate(user_raw_ids, user_keep)
        prompt_msgs[-1][CONTENT] = self.tokenizer.ids_to_text(user_ids)

        # truncate answer
        prompt_raw_ids_len = sys_keep + user_keep
        answer_keep = max(0, min(answer_raw_len, self.max_seq_length - (prompt_tpl_budget + prompt_raw_ids_len + answer_tpl_budget)))
        answer_ids = _truncate(answer_raw_ids, answer_keep)
        msgs[-1][CONTENT] = self.tokenizer.ids_to_text(answer_ids)

        return msgs
        
    def _validate_example(self, sample):
        if sample is None:
            return False
        
        msgs = sample.get(MESSAGES, [])
        if len(msgs) < 2:
            return False
        
        for idx, msg in enumerate(msgs):
            if ROLE not in msg or CONTENT not in msg:
                return False
        
            if msg[ROLE] not in [SYSTEM, USER, ASSISTANT]:
                return False
            
            if msg[ROLE] == SYSTEM and idx != 0:
                return False
            
        if msgs[0][ROLE] == SYSTEM:
            if len(msgs) < 3 or msgs[1][ROLE] != USER:
                return False
            start_idx = 1
        else:
            if msgs[0][ROLE] != USER:
                return False
            start_idx = 0
        
        for i in range(start_idx, len(msgs)):
            expect = USER if (i - start_idx) % 2 == 0 else ASSISTANT
            if msgs[i][ROLE] != expect:
                return False
        
        if msgs[-1][ROLE] != ASSISTANT:
            return False

        return True

    def _ensure_example_format(self, example):
        if isinstance(example, dict) and MESSAGES in example:
            return example
        
        if isinstance(example, list):
            return {MESSAGES: example}
        
        if isinstance(example, dict):
            user_text = example.get("input")
            asst_text = example.get("output")
            metadata = {k: v for k, v in example.items() if k not in ("input", "output")}
            if user_text and asst_text is not None: 
                return {MESSAGES: [{ROLE: USER, CONTENT: user_text}, {ROLE: ASSISTANT, CONTENT: asst_text}], **metadata}

        return None
    
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
            "loss_mask": [0] * example["token_count"] if auto_gen_idx else self._build_loss_mask(example)[1:] + [1.0],
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
