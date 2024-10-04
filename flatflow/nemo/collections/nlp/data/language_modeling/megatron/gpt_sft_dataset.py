# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0rc0/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_dataset.py
# Copyright (c) 2024, The FlatFlow Authors.
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

import math
import re
from typing import List, Mapping, Optional

import datasets
import numpy as np
import torch

# hack to avoid the "not enough disk space" error in some slurm cluster
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.utils import logging

import flatflow.nemo.core.classes

__all__ = ['GPTSFTDataset']

class GPTSFTDataset(flatflow.nemo.core.classes.Dataset):
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
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        global_sample_mapping: bool = False,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
    ):
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
        global_sample_mapping: Whether to shuffle all data together, or shuffle the dataset within each epoch
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        is_test: Whether this dataset is the test split.
        output_original_text (bool): if true, will keep the original text in the output alongside the tokenized ids.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
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
        self.global_sample_mapping = global_sample_mapping
        self.truncation_method = truncation_method
        self.is_test = is_test
        self.output_original_text = output_original_text
        self.ceil_to_power_2 = ceil_to_power_2
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion

        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens

        self._load_dataset()

        # Validate prompt template
        self._maybe_validate_prompt_template()

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()
        self._process_dataset()

    def _load_dataset(self):
        if self.hf_dataset:
            self.indexed_dataset = load_dataset(
                'json',
                data_files=self.file_path,
                cache_dir=self.index_mapping_dir,
                num_proc=self.memmap_workers,
                split='train',
            )
        else:
            self.indexed_dataset = JSONLMemMapDataset(
                dataset_paths=[self.file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=self.index_mapping_dir,
                workers=self.memmap_workers,
            )

    def _maybe_validate_prompt_template(self):
        assert (
            self.prompt_template is not None
        ), f'we need prompt_template to combine contexts and label {self.label_key}'
        # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
        self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        self.prompt_template_keys = re.findall(r'{(.*?)}', self.prompt_template)

        label_placeholder = f'{{{self.label_key}}}'
        assert (
            self.prompt_template[-len(label_placeholder) :] == label_placeholder
        ), f'{label_placeholder} must be at the end of prompt_template.'

        # Legacy checkpoints has self.truncation_fields = ['context'] and self.prompt_template_keys = ['input', 'output']
        if self.prompt_template_keys[0] == 'input' and self.truncation_fields[0] == 'context':
            self.truncation_fields[0] = self.prompt_template_keys[0]

        assert set(self.truncation_fields).issubset(
            self.prompt_template_keys
        ), f'truncation_fields {self.truncation_fields} must in {self.prompt_template_keys}'

    def _build_samples_mapping(self):
        self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        item = self.indexed_dataset[idx]
        input_ids = item['input_ids']
        labels = item['labels']
        seqlen = item['token_count']
        loss_mask = self._build_loss_mask(item) if 0 <= idx else [0] * seqlen
        return {'input_ids': input_ids, 'labels': labels, 'loss_mask': loss_mask, 'seqlen': seqlen}

    def _separate_template(self, prompt_template_values: List[str]):
        """
        Combine contexts and label based on prompt_template into a list of strings and a list of keys.

        Args:
            prompt_template_values (List[str]): the list of context and label strings extrated from jsonl file with prompt_template_keys.

        Returns:
            template_strings (List[str]): separated prompt_template with contexts/label placeholder filled with corresponding strings
            template_strings_keys (List[str]): strings point to placeholder keys or <template>

        Examples:
            prompt_template = 'Context:  {context} Question: {question} Answer: {label}'
            prompt_template_values = ['xxx', 'yyy', 'zzz']

            # tokenizer.space_sensitive = True
            template_strings = ['Context:', '  xxx', ' Question:', ' yyy', ' Answer:', ' zzz']

            # tokenizer.space_sensitive = False
            template_strings = ['Context:', ' xxx', 'Question:', 'yyy', 'Answer:', 'zzz']

            template_strings_keys = ['<template>', 'context', '<template>', 'question', '<template>', 'label']
        """
        placeholders = [f'{{{k}}}' for k in self.prompt_template_keys]

        # placeholder to string
        ph_to_s = {ph: s for ph, s in zip(placeholders, prompt_template_values)}
        # placeholder to key
        ph_to_k = {ph: k for ph, k in zip(placeholders, self.prompt_template_keys)}

        # separate prompt_template based on '<space>{placeholder}'
        # examples:
        #   self.prompt_template = "Context:{context}  Passage: {passage}\n\nQuestion:{question} {label}"
        #   template_with_placeholder_separated = ['Context:', '{context}', '  Passage:', ' {passage}', '\n\nQuestion:', '{question}', ' {label}']
        template_with_placeholder_separated = re.split('( *?{.+?})', self.prompt_template)
        template_with_placeholder_separated = [s for s in template_with_placeholder_separated if len(s) > 0]

        # remove space if we have leading space and tokenizer is not space_sensitive
        # space_sensitive = True : tokenizer.text_to_tokens('A{num_spaces}B') = tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces}B')
        # space_sensitive = False: tokenizer.text_to_tokens('A{num_spaces}B') = tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces-1}B')
        space_sensitive = getattr(self.tokenizer, 'space_sensitive', False)
        template_with_space_reduced = [
            s[1:] if not space_sensitive and s[0] == ' ' else s for s in template_with_placeholder_separated
        ]

        # convert placeholder to the corresponding string (preserve left spaces) and key
        template_strings, template_strings_keys = [], []
        for t in template_with_space_reduced:
            placeholder = t.lstrip(' ')
            left_spaces = ' ' * (len(t) - len(placeholder))
            template_strings.append(left_spaces + ph_to_s.get(placeholder, placeholder))
            template_strings_keys.append(ph_to_k.get(placeholder, '<template>'))

        return template_strings, template_strings_keys

    def _multiple_truncation(self, template_ids: List[List[int]], template_ids_keys: List[str]):
        """
        Calculate total tokens and truncate multiple contexts in truncation_fields.

        Args:
            template_ids (List[List[int]]): the list of separate prompt_template ids.
            template_ids_keys (List[str]): the list of placeholder keys or <template> (used to check key in truncation_fields).

        Returns:
            context_ids (List[int]): all context ids.
            label_ids (List[int]): all label ids.
        """
        context_ids = template_ids[:-1]
        label_ids = template_ids[-1]
        total_ids = (
            self.virtual_tokens
            + sum(len(ids) for ids in context_ids)
            + max(len(label_ids), self.tokens_to_generate)
            + self.add_bos
            + self.add_sep
            + self.add_eos  # Only training need to consider eos token
        )

        if total_ids > self.max_seq_length:
            truncation_length_total = total_ids - self.max_seq_length
            num_fields = len(self.truncation_fields)
            # sorted equal divide length to each field
            # examples:
            #   truncation_length_total = 3
            #   num_fields = 11
            #   truncation_length_list = [3,4,4]
            truncation_length_list = [
                truncation_length_total // num_fields + (1 if i < truncation_length_total % num_fields else 0)
                for i in range(num_fields)[::-1]
            ]

            for i, (ids, key) in enumerate(zip(template_ids, template_ids_keys)):
                if key in self.truncation_fields:
                    truncation_length = truncation_length_list.pop()
                    if len(ids) < truncation_length:
                        logging.warning(f'{key} is not long enough to truncate.')
                        truncation_length = len(ids)

                    if self.truncation_method == 'left':
                        window_offset = truncation_length
                    elif self.truncation_method == 'right':
                        window_offset = 0
                    else:
                        raise ValueError(f'{self.truncation_method} is not supported')

                    window_length = len(ids) - truncation_length
                    template_ids[i] = ids[window_offset : window_offset + window_length]

        context_ids = [i for ids in template_ids[:-1] for i in ids]
        label_ids = template_ids[-1]
        return context_ids, label_ids

    def _truncation(self, ids, expect_length):
        if expect_length == 0:
            return []
        elif self.truncation_method == 'left':
            return ids[-expect_length:]
        elif self.truncation_method == 'right':
            return ids[:expect_length]
        else:
            raise ValueError(f'{self.truncation_method} is not supported')

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        prompt_template_values = []
        for c in self.prompt_template_keys:
            try:
                prompt_template_values.append(example[c].strip(' '))
            except KeyError as e:
                if c == self.label_key and self.is_test:
                    # allow missing label during testing, if user only wants to do inference without calculating metrics
                    prompt_template_values.append("")
                else:
                    raise e

        template_strings, template_strings_keys = self._separate_template(prompt_template_values)
        template_ids = [self.tokenizer.text_to_ids(s) for s in template_strings]
        context_ids, answer_ids = self._multiple_truncation(template_ids, template_strings_keys)

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            context_ids = [self.tokenizer.eos_id] * self.virtual_tokens + context_ids

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]

        input_ids = context_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in self.prompt_template_keys}
        if self.output_original_text:
            for orig_text, text_key in zip(template_strings, template_strings_keys):
                metadata[text_key] = orig_text

        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': len(context_ids),
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'metadata': metadata,
            'token_count': len(input_ids),
        }

        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        if self.ceil_to_power_2:
            # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
            return 2 ** math.ceil(math.log2(n))
        else:
            return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def _build_loss_mask(self, processed_example):
        """Pad input_ids in batch to max batch length while building loss mask"""
        input_ids = processed_example['input_ids']
        answer_start_idx = processed_example['answer_start_idx']
        if self.answer_only_loss:
            loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
        else:
            loss_mask = [1.0] * len(input_ids)

        return loss_mask

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def _process_dataset(self):
        """
        Process the loaded dataset.

        Apply processing of `GPTSFTDataset` first, then further prepare the examples for teacher-forcing
        scheme in advance, not later inside `collate_fn` like `GPTSFTDataset` where you should substract 1
        every time.

        This procedure is highly dependent on `GPTSFTDataset._process_example`, check its update carefully.
        """
        # TODO: multiprocess it for speed-up.
        processed_dataset = []
        for example in self.indexed_dataset:
            example = self._process_example(example)
            example['labels'] = example['input_ids'][1:]
            example['input_ids'] = example['input_ids'][:-1]
            example['token_count'] -= 1
            processed_dataset.append(example)

        self.indexed_dataset = processed_dataset

    def _collate_fn(self, batch):
        """Process & concatenate the batch of samples. Padding is not applied at all."""

        # TODO: use torch.concat() instead of list of np.concatenate()

        input_ids = [np.concatenate([item['input_ids'] for item in batch])]
        labels = [np.concatenate([item['labels'] for item in batch])]
        loss_mask = [np.concatenate([item['loss_mask'] for item in batch])]
        position_ids = [np.concatenate([list(range(item['seqlen'])) for item in batch])]
        token_count = [input_ids[0].shape[0]]

        if input_ids[0].shape[0] != position_ids[0].shape[0]:
            raise ValueError("Dataset problem: input_ids and position_ids lengths don't match")

        # Process cu_seqlens-related values.
        seqlens = np.array([[item['seqlen'] for item in batch]])
        # Here, concatenating -1 at the end is for `cu_seqlens_argmin`-related operations in `MegatronGPTModel`.
        # TBH, padding then removing -1(s) at the end later look unreasonable and error-prone.
        # See a related PR: https://github.com/NVIDIA/NeMo/pull/8108/files
        cu_seqlens = np.concatenate([[[0]], seqlens.cumsum(axis=1), [[-1]]], axis=1)
        cu_seqlens_argmin = np.argmin(cu_seqlens, axis=1, keepdims=True)
        max_seqlen = seqlens.max(keepdims=True)

        processed_batch = {
            'tokens': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(labels),
            'loss_mask': torch.LongTensor(loss_mask),
            'position_ids': torch.LongTensor(position_ids),
            'token_count': token_count,
            'attention_mask': torch.LongTensor([1] * len(input_ids)),  # no attention mask is needed for packed seq
            'cu_seqlens': torch.IntTensor(cu_seqlens),  # cu_seqlens must be in dtype torch.int32
            'cu_seqlens_argmin': torch.IntTensor(cu_seqlens_argmin),
            'max_seqlen': torch.IntTensor(max_seqlen),
        }
        return processed_batch
