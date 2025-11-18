# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py
# Copyright (c) 2024-2025, The FlatFlow Authors.
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

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import get_indexed_dataset_
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import deallocate_indexed_dataset_memory
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig

from flatflow.nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from flatflow.nemo.core.classes import Dataset

__all__ = ["build_train_valid_test_datasets", "GPTDataset"]


def build_dataset(cfg, trainer, data_prefix, data_impl, num_samples, seq_length, seed, skip_warmup, tokenizer, name):
    def _build_dataset(current_data_prefix, current_num_samples):
        delay_data_mmap = cfg.data.get('delay_data_mmap', False)
        indexed_dataset = get_indexed_dataset_(current_data_prefix, data_impl, skip_warmup, delay_data_mmap)
        total_num_of_documents = indexed_dataset.sizes.shape[0]  # type: ignore[has-attr]
        # Print stats about the splits.
        logging.info(' > dataset split:')
        logging.info('     Total {} documents is : {} '.format(name, total_num_of_documents))
        drop_last = True
        if name == "valid":
            drop_last = cfg.data.get("validation_drop_last", True)
        dataset = GPTDataset(
            cfg,
            trainer,
            tokenizer,
            name,
            current_data_prefix,
            np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            indexed_dataset,
            current_num_samples,
            seq_length,
            seed,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)

    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(prefixes)):
            dataset = _build_dataset(prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets)


def build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get('train') is not None
            and data_prefix.get('test') is not None
            and data_prefix.get('validation') is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.data.splits_string is not None:
            logging.warning(cfg.data.splits_string + " ignored since data prefix is of type dictionary.")
        train_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["train"],
            data_impl,
            int(train_valid_test_num_samples[0]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "train",
        )
        validation_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["validation"],
            data_impl,
            int(train_valid_test_num_samples[1]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "valid",
        )
        test_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["test"],
            data_impl,
            int(train_valid_test_num_samples[2]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "test",
        )
        return train_ds, validation_ds, test_ds

    else:
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cfg,
                trainer,
                data_prefix[0],
                data_impl,
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cfg,
                trainer,
                prefixes[i],
                data_impl,
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    delay_data_mmap = cfg.data.get('delay_data_mmap', False)
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup, delay_data_mmap)

    total_num_of_documents = indexed_dataset.sizes.shape[0]  # type: ignore[has-attr]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(' > dataset split:')
    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "valid":
                drop_last = cfg.data.get("validation_drop_last", True)
            dataset = GPTDataset(
                cfg,
                trainer,
                tokenizer,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')
    return (train_dataset, valid_dataset, test_dataset)


def count_file_path(prefix_path: str) -> str:
    return prefix_path + "_cnt.npy"


class GPTDataset(Dataset):
    """Dataset for GPT pre-training."""

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        drop_last=True,
    ) -> None:
        super().__init__()
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.drop_last = drop_last
        self.seq_length = seq_length
        self.get_attention_mask_from_fusion = cfg.get("get_attention_mask_from_fusion", True)

        assert 0 <= np.min(documents)
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.reset_position_ids = cfg.data.get("reset_position_ids", False)
        self.reset_attention_mask = cfg.data.get("reset_attention_mask", False)
        self.eod_mask_loss = cfg.data.get("eod_mask_loss", False)
        self.create_inputs = any([self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss])
        self.cached_inputs = False
        self.eos_id = tokenizer.eos_id
        self.no_seqlen_plus_one_input_tokens = cfg.data.get("no_seqlen_plus_one_input_tokens", False)
        self.add_extra_token = 1
        if self.no_seqlen_plus_one_input_tokens:
            self.add_extra_token = 0
        self.shuffle_documents = cfg.data.get("shuffle_documents", True)
        self.exchange_indices_distributed = cfg.data.get("exchange_indices_distributed", False)

        deallocate_indexed_dataset_memory(self.indexed_dataset)

        self._sizes = np.load(count_file_path(data_prefix))
        self._sizes -= 1

    def create_data_mmap(self):
        self.indexed_dataset.create_data_mmap()

    def __getitem__(self, idx):
        sample = self.indexed_dataset.get(idx)
        text = torch.from_numpy(sample.astype(np.int64))

        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()

        # Derived from nemo...megatron/gpt_dataset.py:_create_ltor_masks_and_position_ids
        loss_mask = torch.ones(len(tokens), dtype=torch.float)
        if self.eod_mask_loss:
            loss_mask[tokens == self.eos_id] = 0.0

        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        # Negative index comes when we pad the last batch in MegatronPretrainingBatchSampler
        if idx < 0:
            logging.debug("Got negative index. Masking loss from this sample")
            loss_mask = torch.zeros_like(loss_mask)

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "seqlen": len(tokens),
        }

    def __len__(self):
        return len(self.indexed_dataset)

    def __sizeof__(self, idx):
        return self._sizes[idx]

    def collate_fn(self, batch):
        tokens = np.concatenate([item["tokens"].numpy() for item in batch])
        labels = np.concatenate([item["labels"].numpy() for item in batch])
        loss_mask = np.concatenate([item["loss_mask"].numpy() for item in batch])
        position_ids = np.concatenate([list(range(item["seqlen"])) for item in batch])

        # Convert token_count to tensor instead of keeping as int
        token_count = torch.LongTensor([tokens.shape[0]])

        assert tokens.shape[0] == position_ids.shape[0]

        seqlens = np.array([item["seqlen"] for item in batch])
        cu_seqlens = np.concatenate([[0], seqlens.cumsum(), [-1]])
        cu_seqlens_argmin = np.argmin(cu_seqlens, keepdims=True)
        max_seqlen = seqlens.max(keepdims=True)

        return {
            "tokens": torch.LongTensor(tokens).unsqueeze(0),
            "labels": torch.LongTensor(labels).unsqueeze(0),
            "loss_mask": torch.FloatTensor(loss_mask).unsqueeze(0),
            "position_ids": torch.LongTensor(position_ids).unsqueeze(0),
            "token_count": token_count,
            "attention_mask": torch.LongTensor([1]),
            "cu_seqlens": torch.IntTensor(cu_seqlens).unsqueeze(0),
            "cu_seqlens_argmin": torch.IntTensor(cu_seqlens_argmin).unsqueeze(0),
            "max_seqlen": torch.IntTensor(max_seqlen).unsqueeze(0),
        }
