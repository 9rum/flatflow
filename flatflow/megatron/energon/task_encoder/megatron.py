import json

import numpy
import torch
from megatron.core.datasets.gpt_dataset import (
    GPTDatasetConfig,
    _get_ltor_masks_and_position_ids,
)
from megatron.energon import DefaultTaskEncoder, TextSample
from torch.utils.data import default_collate

__all__ = ["MegatronTaskEncoder"]


class MegatronTaskEncoder(DefaultTaskEncoder):
    def __init__(self, config: GPTDatasetConfig):
        super().__init__(
            encoded_sample_type=dict[str, torch.Tensor],
            raw_batch_type=dict[str, torch.Tensor],
            batch_type=dict[str, torch.Tensor],
        )
        self.config = config
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        try:
            self._pad_token_id = self.config.tokenizer.pad  # type: ignore[has-attr]
        except Exception:
            self._pad_token_id = -1

    def encode_sample(self, sample: TextSample) -> dict[str, torch.Tensor]:
        text = torch.from_numpy(numpy.array(json.loads(sample.text), dtype=numpy.int64))
        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,  # type: ignore[has-attr]
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        loss_mask[labels == self._pad_token_id] = 0.0
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def batch(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return default_collate(samples)
