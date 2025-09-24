# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py
# Copyright (c) 2025, The FlatFlow Authors.
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import (
    BlendableDataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    GPTDataset,
    _build_index_mappings,
    _create_ltor_masks_and_position_ids,
    get_indexed_dataset_,
)
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import (
    deallocate_indexed_dataset_memory,
)
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig

__all__ = ["build_obfd_datasets"]


class OBFDDataset(GPTDataset):
    """Dataset for `Fewer Truncations Improve Language Modeling`.

    You should add the following options to the `data` section in the config file:
      - use_obfd (bool)
      - obfd_token_data_prefix (list)
      - obfd_label_data_prefix (list)
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        token_data_prefix,
        label_data_prefix,
        documents,
        indexed_token_dataset,
        indexed_label_dataset,
        num_samples,
        seq_length,
        seed,
        drop_last=True,
    ):
        super().__init__(
            cfg,
            trainer,
            tokenizer,
            name,
            token_data_prefix,
            documents,
            indexed_token_dataset,
            num_samples,
            seq_length,
            seed,
            drop_last,
        )
        self.indexed_label_dataset = indexed_label_dataset
        assert np.max(documents) < indexed_label_dataset.sizes.shape[0]
        self.add_extra_token = 0

        # Build index mappings.
        self.label_doc_idx, self.label_sample_idx, self.label_shuffle_idx = (
            _build_index_mappings(
                self.name,
                label_data_prefix,
                documents,
                self.indexed_label_dataset.sizes,
                num_samples,
                seq_length,
                seed,
                index_mapping_dir=self.index_mapping_dir,
                drop_last=drop_last,
                add_extra_token=self.add_extra_token,
                shuffle_documents=self.shuffle_documents,
                exchange_indices_distributed=self.exchange_indices_distributed,
            )
        )
        deallocate_indexed_dataset_memory(self.indexed_label_dataset)

    def create_data_mmap(self):
        super().create_data_mmap()
        self.indexed_label_dataset.create_data_mmap()

    def _get_label(self, idx):
        # Get the shuffled index.
        idx = self.label_shuffle_idx[idx]

        # Start and end documents and offsets.
        doc_index_f = self.label_sample_idx[idx][0]
        doc_index_l = self.label_sample_idx[idx + 1][0]
        offset_f = self.label_sample_idx[idx][1]
        offset_l = self.label_sample_idx[idx + 1][1]

        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_label_dataset.get(
                self.label_doc_idx[doc_index_f],
                offset=offset_f,
                length=offset_l - offset_f + self.add_extra_token,
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [
                self.indexed_label_dataset.get(
                    self.label_doc_idx[doc_index_f], offset=offset_f
                )
            ]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(
                    self.indexed_label_dataset.get(self.label_doc_idx[i])
                )
            # And finally add the relevant portion of last document.
            sample_list.append(
                self.indexed_label_dataset.get(
                    self.label_doc_idx[doc_index_l],
                    length=offset_l + self.add_extra_token,
                )
            )
            sample = np.concatenate(sample_list)
        if len(sample) != (self.seq_length + self.add_extra_token):
            logging.info(
                f" > WARNING: Got sample of length: {len(sample)}"
                f" for sequence length={self.seq_length + self.add_extra_token},"
                " padding the sample to match sequence length"
            )
            sample = np.array(sample, dtype=np.int64)
            sample = np.pad(
                sample,
                (0, self.seq_length + self.add_extra_token - len(sample)),
                mode="constant",
                constant_values=-1,
            )
        return sample.astype(np.int64)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self._get_text(idx))
        labels = torch.from_numpy(self._get_label(idx))

        if self.create_inputs or not self.cached_inputs:
            attention_mask, loss_mask, position_ids = (
                _create_ltor_masks_and_position_ids(
                    tokens,
                    self.eos_id,
                    self.reset_position_ids,
                    self.reset_attention_mask,
                    self.eod_mask_loss,
                )
            )
            if not self.create_inputs:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.cached_inputs = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids
        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        # Negative index comes when we pad the last batch in
        # MegatronPretrainingBatchSampler. We make the loss_mask zero to mask out loss
        # from these samples.
        if idx < 0:
            logging.debug("Got negative index. Masking loss from this sample")
            loss_mask = torch.zeros_like(loss_mask)

        if self.get_attention_mask_from_fusion:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }


def _build_train_valid_test_datasets(
    cfg,
    trainer,
    token_data_prefix,
    label_data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    delay_data_mmap = cfg.data.get("delay_data_mmap", False)
    indexed_token_dataset = get_indexed_dataset_(
        token_data_prefix, data_impl, skip_warmup, delay_data_mmap
    )
    indexed_label_dataset = get_indexed_dataset_(
        label_data_prefix, data_impl, skip_warmup, delay_data_mmap
    )

    total_num_of_documents = indexed_token_dataset.sizes.shape[0]  # type: ignore
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    def build_dataset(index, name):
        if splits[index] < splits[index + 1]:
            documents = np.arange(
                start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32
            )
            drop_last = True
            if name == "valid":
                drop_last = cfg.data.get("validation_drop_last", True)
            return OBFDDataset(
                cfg,
                trainer,
                tokenizer,
                name,
                token_data_prefix,
                label_data_prefix,
                documents,
                indexed_token_dataset,
                indexed_label_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                drop_last=drop_last,
            )
        return None

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")
    return (train_dataset, valid_dataset, test_dataset)


def build_train_valid_test_datasets(
    cfg,
    trainer,
    token_data_prefix,
    label_data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    assert data_impl != "mock"
    assert not isinstance(token_data_prefix, DictConfig)
    assert not isinstance(label_data_prefix, DictConfig)
    assert len(token_data_prefix) == len(label_data_prefix)

    # Single dataset.
    if len(token_data_prefix) == 1:
        return _build_train_valid_test_datasets(
            cfg,
            trainer,
            token_data_prefix[0],
            label_data_prefix[0],
            data_impl,
            splits_string,
            train_valid_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
        )

    # Blending dataset.
    token_output = get_datasets_weights_and_num_samples(
        token_data_prefix, train_valid_test_num_samples
    )
    token_prefixes, weights, datasets_train_valid_test_num_samples = token_output

    label_output = get_datasets_weights_and_num_samples(
        label_data_prefix, train_valid_test_num_samples
    )
    label_prefixes, _, _ = label_output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(token_prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            cfg,
            trainer,
            token_prefixes[i],
            label_prefixes[i],
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

    train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

    # Blend.
    blending_train_dataset = None
    blending_valid_dataset = None
    blending_test_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights, train_n)
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n)
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights, test_n)
    return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def build_obfd_datasets(
    cfg,
    trainer,
    token_data_prefix,
    label_data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    return build_train_valid_test_datasets(
        cfg,
        trainer,
        token_data_prefix,
        label_data_prefix,
        data_impl,
        splits_string,
        train_valid_test_num_samples,
        seq_length,
        seed,
        skip_warmup,
        tokenizer,
    )
