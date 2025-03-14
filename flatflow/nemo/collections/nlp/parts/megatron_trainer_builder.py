# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/parts/megatron_trainer_builder.py
# Copyright (c) 2025, The FlatFlow Authors.
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

import sys
from typing import Optional, Union

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary

from nemo.utils import logging

import nemo.collections.nlp.parts.megatron_trainer_builder

from nemo.collections.nlp.parts.nlp_overrides import (
    NLPFSDPStrategy,
    GradScaler,
    NLPDDPStrategyNotebook,
    NLPDDPStrategy,
)

from flatflow.megatron.core.bpipe import bpipe_state

class MegatronTrainerBuilder(nemo.collections.nlp.parts.megatron_trainer_builder.MegatronTrainerBuilder):
    """
    Builder type to hide complex configuration of PTL Trainers for Megatron LLM models.
    Can be extended to change behavior for a specific model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _training_strategy(self) -> Union[NLPDDPStrategy, NLPFSDPStrategy]:
        """
        Returns a DDP or a FSDP strategy passed to Trainer.strategy.
        """
        # check interactive environment
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)
        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            logging.info("Detected interactive environment, using NLPDDPStrategyNotebook")
            return NLPDDPStrategyNotebook(
                no_ddp_communication_hook=True,
                find_unused_parameters=False,
            )

        assert not self.cfg.model.get("fsdp", False), "FSDP is not supported in BPipe yet"

        # Set BPipe options before setting DDP strategy to ensure proper 
        # pipeline parallel (PP) behavior. Override PP-related functions 
        # to integrate BPipe's memory-balanced execution.
        use_bpipe = self.cfg.model.get("use_bpipe", False)
        bpipe_state.set_bpipe_option(use_bpipe)

        return NLPDDPStrategy(
            no_ddp_communication_hook=True,
            gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
            nccl_communicator_config_path=self.cfg.model.get('nccl_communicator_config_path', None),
            sharp=self.cfg.model.get('sharp', False),
            dist_ckpt_parallel_save=self.cfg.model.get('dist_ckpt_parallel_dist_opt', True),
        )

        
class MegatronLMPPTrainerBuilder(MegatronTrainerBuilder):
    """Builder for scripts where grad scaler is turned off for pipeline parallel LM model. E.g. PEFT tuning scripts"""

    def _grad_scaler(self) -> GradScaler:
        return GradScaler(
            init_scale=self.cfg.model.get("native_amp_init_scale", 2**32),
            growth_interval=self.cfg.model.get("native_amp_growth_interval", 1000),
            hysteresis=self.cfg.model.get("hysteresis", 2),
            enabled=False if self.cfg.model.pipeline_model_parallel_size > 1 else True,
        )
