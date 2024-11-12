# Copyright (c) 2024, The FlatFlow Authors.
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

from collections import defaultdict
from typing import Any, Optional
import time
try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

class LatencyProfiler:
    def __init__(self, rank, layer_idx, num_layers, hook_handles=[]):
        self.start_times = {}
        self.forward_times = defaultdict(float)
        self.buffer = defaultdict(float)
        self.rank = rank
        self.world_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.current_batch_id = 0
        self.hook_handles = hook_handles
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        
    def _generate_batch_key(self, microbatch_id: int) -> str:
        return f"dp{parallel_state.get_data_parallel_rank()}_pp{parallel_state.get_pipeline_model_parallel_rank()}_tp{parallel_state.get_tensor_model_parallel_rank()}_batch{microbatch_id}"

    def update_microbatch_id(self):
        self.current_batch_id += 1

    def record_start(self, module: Any, input: Any) -> None:
        batch_key = self._generate_batch_key(self.current_batch_id)
        self.start_times[batch_key] = time.perf_counter()

    def is_last_layer(self) -> bool:
        """Check if current layer is the last transformer layer in the pipeline stage"""
        layers_per_stage = self.num_layers // self.world_size
        start_layer_idx = parallel_state.get_pipeline_model_parallel_rank() * layers_per_stage
        end_layer_idx = start_layer_idx + layers_per_stage - 1
        return self.layer_idx == end_layer_idx

    def record_end(self, module: Any, input: Any, output: Any) -> None:
        batch_key = self._generate_batch_key(self.current_batch_id)
        if batch_key in self.start_times:
            elapsed_time = (time.perf_counter() - self.start_times[batch_key]) * 1000
            self.buffer[batch_key] += elapsed_time
        
        # Only record final timing if we're in the last stage AND this is the last layer
        if parallel_state.is_pipeline_last_stage() and self.is_last_layer():
            self.forward_times[batch_key] = self.buffer[batch_key]

    def get_timing_data(self):
        return self.forward_times