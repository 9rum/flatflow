# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py
# Copyright (c) 2024-2025, The FlatFlow Authors.
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

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset as NeMoGPTSFTChatDataset

from flatflow.nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset

__all__ = ["GPTSFTChatDataset"]


class GPTSFTChatDataset(GPTSFTDataset, NeMoGPTSFTChatDataset):
    """
    It inherits and utilizes methods from parent classes as below:

    flatflow.nemo...GPTSFTDataset to be compatible with FlatFlow scheduler.
    - `__init__`
    - `__getitem__`
    - `__sizeof__`
    - `_collaten_fn`
    - `collate_fn`

    NeMoGPTSFTChatDataset to be compatible with NeMo framework's data format
    - `_process_example`
    """

    def _build_loss_mask(self, processed_example):
        """Build loss mask using the `mask` field of processed examples.

        The original logic is found in `NeMoGPTSFTChatDataset.collate_fn`.
        """
        loss_mask = processed_example['mask'][1:].tolist()
        return loss_mask
